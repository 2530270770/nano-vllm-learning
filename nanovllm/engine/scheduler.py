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
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> ScheduleOutput:
        decode_seqs = []
        prefill_seqs = []
        num_batched_tokens = 0

        # =========================
        # 1. Decode Scheduling
        # =========================
        max_decode_seqs = min(
            self.max_num_decode_seqs,
            self.max_num_seqs,
            self.max_num_batched_tokens,
        )

        while self.running and len(decode_seqs) < max_decode_seqs:
            seq = self.running.popleft()

            # 确保当前 Decode 序列可以追加 KV Cache
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)

                decode_seqs.append(seq)
                num_batched_tokens += 1

        # 将本轮选中的 Decode 序列放回 running 队列
        self.running.extendleft(reversed(decode_seqs))

        # =========================
        # 2. Prefill Scheduling
        # =========================
        remaining = self.max_num_batched_tokens - num_batched_tokens
        available_slots = self.max_num_seqs - len(decode_seqs)

        # 本轮最多处理多少个不同的 Prefill 请求
        # 使用初始候选数量，避免同一个未完成请求被重复调度
        num_candidates = min(
            len(self.waiting),
            available_slots,
            remaining,
        )

        for i in range(num_candidates):
            if not self.waiting:
                break

            seq = self.waiting.popleft()

            # 获取当前请求尚未 Prefill 的 Token 数量
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)

                if num_cached_blocks == -1:
                    # 当前请求无法分配 KV Cache，恢复队列顺序并停止
                    self.waiting.appendleft(seq)
                    break

                num_tokens = (
                    seq.num_tokens - num_cached_blocks * self.block_size
                )
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # 公平分配剩余 Token Budget
            candidates_left = num_candidates - i
            fair_share = (
                remaining + candidates_left - 1
            ) // candidates_left

            chunk_tokens = min(num_tokens, fair_share)

            if chunk_tokens <= 0:
                self.waiting.appendleft(seq)
                break

            # 首次调度时分配 KV Cache
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            seq.num_scheduled_tokens = chunk_tokens
            seq.is_prefill = True
            prefill_seqs.append(seq)
            
            num_batched_tokens += chunk_tokens
            remaining -= chunk_tokens

            # Prefill 完成，转入 running
            if seq.num_cached_tokens + chunk_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
            else:
                # 尚未完成，放到 waiting 队列尾部
                # 下一轮继续执行剩余 Chunk
                self.waiting.append(seq)

        # =========================
        # 3. Build ScheduleOutput
        # =========================
        return ScheduleOutput(
            decode_seqs=decode_seqs,
            prefill_seqs=prefill_seqs,
            decode_tokens=len(decode_seqs),
            prefill_tokens=sum(
                seq.num_scheduled_tokens for seq in prefill_seqs
            ),
        )


    def preempt(self, seq: Sequence):
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
