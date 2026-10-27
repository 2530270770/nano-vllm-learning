from collections import deque
from typing import Callable
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config, event_callback: Callable[[dict], None] | None = None,):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.event_callback = event_callback
        self.step_id = 0

    def emit_event(self, event: dict):
        if self.event_callback is not None:
            self.event_callback(event)

    def emit_schedule_event(
        self,
        scheduled_seqs: list[Sequence],
        is_prefill: bool,
        step: int,
    ):
        self.emit_event({
            "event": "schedule",
            "step": step,
            "phase": "prefill" if is_prefill else "decode",
            "waiting": len(self.waiting),
            "running": len(self.running),
            "scheduled_seq_ids": [seq.seq_id for seq in scheduled_seqs],
            "batched_tokens": sum(
                seq.num_scheduled_tokens for seq in scheduled_seqs
            ),
            "free_blocks": len(self.block_manager.free_block_ids),
            "total_blocks": (
                len(self.block_manager.free_block_ids)
                + len(self.block_manager.used_block_ids)
            ),
        })

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

        self.emit_event({
            "event": "request_added",
            "step": self.step_id,
            "seq_id": seq.seq_id,
            "waiting": len(self.waiting),
            "running": len(self.running),
            "free_blocks": len(self.block_manager.free_block_ids),
            "total_blocks": (
                len(self.block_manager.free_block_ids)
                + len(self.block_manager.used_block_ids)
            ),
        })

    def schedule(self) -> tuple[list[Sequence], bool]:

        self.emit_event({
            "event": "schedule_start",
            "step": self.step_id,
            "waiting": len(self.waiting),
            "running": len(self.running),
            "free_blocks": len(self.block_manager.free_block_ids),
            "total_blocks": (
                len(self.block_manager.free_block_ids)
                + len(self.block_manager.used_block_ids)
            ),
        })

        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            self.emit_schedule_event(
                scheduled_seqs, is_prefill=True, step=self.step_id
            )
            self.step_id += 1
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
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
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))

        self.emit_schedule_event(
            scheduled_seqs, is_prefill=False, step=self.step_id
        )
        self.step_id += 1

        return scheduled_seqs, False

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
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

        self.emit_event({
            "event": "postprocess",
            "step": self.step_id - 1, # 因为schedule返回时已经将stepid + 1,这里减一表示是同一个step
            "waiting": len(self.waiting),
            "running": len(self.running),
            "finished_seq_ids": [
                seq.seq_id for seq in seqs if seq.is_finished
            ],
            "free_blocks": len(self.block_manager.free_block_ids),
            "total_blocks": (
                len(self.block_manager.free_block_ids)
                + len(self.block_manager.used_block_ids)
            ),
        })
