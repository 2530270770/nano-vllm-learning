from collections import deque
from dataclasses import dataclass
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

@dataclass
class ScheduleOutput:
    decode_seqs: list[Sequence]
    prefill_seqs: list[Sequence]
    decode_tokens: int # = len(decode_seqs)
    prefill_tokens: int

class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_decode_seqs = config.max_num_decode_seqs
        self.max_num_prefill_seqs = getattr(
            config,
            "max_num_prefill_seqs",
            None,
        )
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

        self.enable_prefill_aging = getattr(
            config,
            "enable_prefill_aging",
            False,
        )
        self.schedule_step = 0
        self.last_schedule_stats = {}
        self.total_kv_allocation_failures = 0
        self.total_decode_preemptions = 0

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> ScheduleOutput:
        decode_seqs = []
        prefill_seqs = []
        step_id = self.schedule_step
        self.schedule_step += 1
        waiting_at_start = len(self.waiting)
        running_at_start = len(self.running)
        kv_allocation_failures = []
        decode_preemptions_before = self.total_decode_preemptions

        # ==================================================
        # 1. Decode Scheduling
        # ==================================================

        # Decode 独立上限，同时受到共享请求数和 Token Budget 限制
        max_decode_seqs = min(
            self.max_num_decode_seqs,
            self.max_num_seqs,
            self.max_num_batched_tokens,
        )

        while self.running and len(decode_seqs) < max_decode_seqs:
            # 取出一个 Decode 候选请求
            seq = self.running.popleft()

            # 检查是否有足够的 KV Cache Block 用于追加
            while not self.block_manager.can_append(seq):

                # 优先抢占尚未被选入 Decode Batch 的请求
                if self.running:
                    victim = self.running.pop()
                    self.preempt(victim)

                else:
                    # 没有其他请求可以抢占，只能抢占当前请求
                    self.preempt(seq)
                    seq = None
                    break

            # 当前请求已经被抢占，继续检查其他请求
            if seq is None:
                continue

            # 预留 KV Cache Block
            self.block_manager.may_append(seq)

            seq.num_scheduled_tokens = 1
            seq.is_prefill = False

            decode_seqs.append(seq)

        # 将本轮选中的 Decode 请求放回 running 队列
        # 保持原来的请求顺序
        self.running.extendleft(reversed(decode_seqs))

        decode_tokens = len(decode_seqs)

        # ==================================================
        # 2. Calculate Remaining Budget
        # ==================================================

        remaining_tokens = self.max_num_batched_tokens - decode_tokens
        available_slots = self.max_num_seqs - len(decode_seqs)

        # 本轮最多能够调度的 Prefill 请求数量
        # 每个 Prefill 请求至少需要分配 1 个 Token
        max_prefill_seqs = min(
            available_slots,
            remaining_tokens,
        )

        if self.max_num_prefill_seqs is not None:
            max_prefill_seqs = min(
                max_prefill_seqs,
                self.max_num_prefill_seqs,
            )

        # ==================================================
        # 3. Collect Prefill Candidates
        # ==================================================

        candidates = []

        # 记录本轮开始时 Waiting 队列的长度
        waiting_count = len(self.waiting)

        waiting_candidates = []

        # ----------------------------------------------
        # 3.1 收集本轮开始时所有 Waiting 请求
        # ----------------------------------------------

        for _ in range(waiting_count):
            seq = self.waiting.popleft()

            # 累计等待步数始终记录；Aging 开关只影响排序，不影响指标口径。
            seq.prefill_total_wait_steps += 1
            if self.enable_prefill_aging:
                # 连续等待年龄仅在 Aging 启用时参与排序。
                seq.prefill_wait_steps += 1

            waiting_candidates.append(seq)

        # ----------------------------------------------
        # 3.2 按等待时间排序
        # ----------------------------------------------

        if self.enable_prefill_aging:
            # 等待时间越长，越优先获得 Prefill 调度名额
            #
            # Python 排序是稳定排序：
            # 等待时间相同时，保留原来的队列顺序。
            waiting_candidates.sort(
                key=lambda seq: seq.prefill_wait_steps,
                reverse=True,
            )

        # ----------------------------------------------
        # 3.3 选择 Prefill 候选请求
        # ----------------------------------------------

        for seq in waiting_candidates:

            # 已经达到本轮 Prefill 并发上限
            if len(candidates) >= max_prefill_seqs:
                self.waiting.append(seq)
                continue

            # ------------------------------------------
            # 首次调度：分配 KV Cache
            # ------------------------------------------

            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)

                # KV Cache 不足：
                # 跳过当前请求，继续检查后面的请求
                if num_cached_blocks == -1:
                    kv_allocation_failures.append(seq.seq_id)
                    self.total_kv_allocation_failures += 1
                    self.waiting.append(seq)
                    continue

                self.block_manager.allocate(seq, num_cached_blocks)

            # ------------------------------------------
            # 计算剩余 Prefill Token
            # ------------------------------------------

            remaining_seq_tokens = (
                seq.num_tokens - seq.num_cached_tokens
            )

            # 理论上 Waiting 中不应该有已经完成 Prefill 的请求
            if remaining_seq_tokens <= 0:
                seq.status = SequenceStatus.RUNNING
                seq.prefill_wait_steps = 0
                self.running.append(seq)
                continue

            candidates.append((seq, remaining_seq_tokens))
        # ==================================================
        # 4. Integer Water-filling
        # ==================================================

        if candidates and remaining_tokens > 0:

            demands = [
                remaining for _, remaining in candidates
            ]

            # ----------------------------------------------
            # 4.1 二分查找最大均分水位
            # ----------------------------------------------

            low = 0
            high = max(demands)

            while low < high:
                mid = (low + high + 1) // 2

                allocated = sum(
                    min(demand, mid)
                    for demand in demands
                )

                if allocated <= remaining_tokens:
                    low = mid
                else:
                    high = mid - 1

            # low 即最大整数水位
            water_level = low

            # ----------------------------------------------
            # 4.2 按水位分配 Token
            # ----------------------------------------------

            allocations = [
                min(demand, water_level)
                for demand in demands
            ]

            used_tokens = sum(allocations)
            leftover_tokens = remaining_tokens - used_tokens

            # ----------------------------------------------
            # 4.3 分配剩余 Token
            # ----------------------------------------------

            # 给尚未满足需求的请求依次多分配 1 个 Token
            for i, demand in enumerate(demands):

                if leftover_tokens <= 0:
                    break

                if demand > water_level:
                    allocations[i] += 1
                    leftover_tokens -= 1

            # ==================================================
            # 5. Build Prefill Batch
            # ==================================================

            for (seq, remaining_seq_tokens), chunk_tokens in zip(
                candidates, allocations
            ):

                if chunk_tokens <= 0:
                    # 理论上不会进入这里，保留保护逻辑
                    self.waiting.append(seq)
                    continue

                seq.num_scheduled_tokens = chunk_tokens
                seq.is_prefill = True

                # 请求已经获得本轮 Prefill 调度机会
                seq.prefill_wait_steps = 0

                prefill_seqs.append(seq)

                # 判断本轮 Chunk 是否完成整个 Prefill
                if (
                    seq.num_cached_tokens + chunk_tokens
                    >= seq.num_tokens
                ):
                    seq.status = SequenceStatus.RUNNING
                    self.running.append(seq)

                else:
                    # 尚未完成，放回 Waiting 队列尾部
                    seq.status = SequenceStatus.WAITING
                    self.waiting.append(seq)

        # ==================================================
        # 6. Build ScheduleOutput
        # ==================================================

        prefill_tokens = sum(
            seq.num_scheduled_tokens
            for seq in prefill_seqs
        )

        allocated_tokens = decode_tokens + prefill_tokens
        free_blocks = len(self.block_manager.free_block_ids)
        total_blocks = len(self.block_manager.blocks)
        self.last_schedule_stats = {
            "step": step_id,
            "decode_limit": max_decode_seqs,
            "decode_count": len(decode_seqs),
            "decode_tokens": decode_tokens,
            "prefill_limit": max_prefill_seqs,
            "prefill_candidates": len(candidates),
            "prefill_candidate_seq_ids": [seq.seq_id for seq, _ in candidates],
            "prefill_demands": [demand for _, demand in candidates],
            "prefill_allocations": [seq.num_scheduled_tokens for seq in prefill_seqs],
            "prefill_seq_ids": [seq.seq_id for seq in prefill_seqs],
            "prefill_total_wait_steps": [seq.prefill_total_wait_steps for seq in prefill_seqs],
            "prefill_max_total_wait_steps": max((seq.prefill_total_wait_steps for seq in prefill_seqs), default=0),
            "prefill_tokens": prefill_tokens,
            "token_budget": self.max_num_batched_tokens,
            "used_tokens": allocated_tokens,
            "unused_tokens": self.max_num_batched_tokens - allocated_tokens,
            "budget_utilization": allocated_tokens / self.max_num_batched_tokens,
            "kv_allocation_failures": kv_allocation_failures,
            "total_kv_allocation_failures": self.total_kv_allocation_failures,
            "free_kv_blocks": free_blocks,
            "used_kv_blocks": total_blocks - free_blocks,
            "total_kv_blocks": total_blocks,
            "waiting_at_start": waiting_at_start,
            "running_at_start": running_at_start,
            "waiting_at_end": len(self.waiting),
            "running_at_end": len(self.running),
            "decode_preemptions": self.total_decode_preemptions - decode_preemptions_before,
            "total_decode_preemptions": self.total_decode_preemptions,
        }

        return ScheduleOutput(
            decode_seqs=decode_seqs,
            prefill_seqs=prefill_seqs,
            decode_tokens=decode_tokens,
            prefill_tokens=prefill_tokens,
        )


    def preempt(self, seq: Sequence):
        self.total_decode_preemptions += 1
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens: # chunk prefill的判断
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
