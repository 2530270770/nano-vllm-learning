
from types import SimpleNamespace

import pytest

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import ScheduleOutput


class FakeScheduler:
    """模拟 Scheduler，只记录调用，不执行真实调度。"""

    def __init__(self, schedule_output, events):
        self.schedule_output = schedule_output
        self.events = events

    def schedule(self):
        self.events.append("schedule")
        return self.schedule_output

    def postprocess(self, seqs, token_ids, is_prefill):
        phase = "prefill" if is_prefill else "decode"

        self.events.append(
            ("postprocess", phase, [seq.seq_id for seq in seqs])
        )

        for seq, token_id in zip(seqs, token_ids):
            seq.completion_token_ids.append(token_id)
            seq.is_finished = True


class FakeModelRunner:
    """模拟 ModelRunner.call()。"""

    def __init__(self, events):
        self.events = events

    def call(self, method_name, seqs):
        self.events.append(
            (method_name, [seq.seq_id for seq in seqs])
        )

        if method_name == "run_decode":
            return [101 + i for i in range(len(seqs))]

        if method_name == "run_prefill":
            return [201 + i for i in range(len(seqs))]

        raise ValueError(f"Unexpected method: {method_name}")


def make_seq(seq_id):
    """创建一个简单的模拟 Sequence。"""
    return SimpleNamespace(
        seq_id=seq_id,
        completion_token_ids=[],
        is_finished=False,
    )


@pytest.fixture
def make_engine():
    """创建不初始化 CUDA、模型和 Tokenizer 的 LLMEngine。"""

    def _make_engine(schedule_output):
        events = []

        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = FakeScheduler(schedule_output, events)
        engine.model_runner = FakeModelRunner(events)

        return engine, events

    return _make_engine


def test_step_mixed_decode_and_prefill(make_engine):
    """混合批次必须先 Decode，再 Prefill。"""

    decode_seqs = [make_seq(1), make_seq(2)]
    prefill_seqs = [make_seq(3)]

    schedule_output = ScheduleOutput(
        decode_seqs=decode_seqs,
        prefill_seqs=prefill_seqs,
        decode_tokens=2,
        prefill_tokens=128,
    )

    engine, events = make_engine(schedule_output)

    outputs, decode_tokens, prefill_tokens = engine.step()

    # 验证完整执行顺序
    assert events == [
        "schedule",
        ("run_decode", [1, 2]),
        ("postprocess", "decode", [1, 2]),
        ("run_prefill", [3]),
        ("postprocess", "prefill", [3]),
    ]

    # 验证返回的完成请求
    assert outputs == [
        (1, [101]),
        (2, [102]),
        (3, [201]),
    ]

    # 验证两种计算的 Token 数量
    assert decode_tokens == 2
    assert prefill_tokens == 128


def test_step_decode_only(make_engine):
    """只有 Decode 请求时，不应该执行 Prefill。"""

    decode_seqs = [make_seq(1)]

    schedule_output = ScheduleOutput(
        decode_seqs=decode_seqs,
        prefill_seqs=[],
        decode_tokens=1,
        prefill_tokens=0,
    )

    engine, events = make_engine(schedule_output)

    outputs, decode_tokens, prefill_tokens = engine.step()

    assert events == [
        "schedule",
        ("run_decode", [1]),
        ("postprocess", "decode", [1]),
    ]

    assert outputs == [(1, [101])]
    assert decode_tokens == 1
    assert prefill_tokens == 0


def test_step_prefill_only(make_engine):
    """只有 Prefill 请求时，不应该执行 Decode。"""

    prefill_seqs = [make_seq(2), make_seq(3)]

    schedule_output = ScheduleOutput(
        decode_seqs=[],
        prefill_seqs=prefill_seqs,
        decode_tokens=0,
        prefill_tokens=256,
    )

    engine, events = make_engine(schedule_output)

    outputs, decode_tokens, prefill_tokens = engine.step()

    assert events == [
        "schedule",
        ("run_prefill", [2, 3]),
        ("postprocess", "prefill", [2, 3]),
    ]

    assert outputs == [
        (2, [201]),
        (3, [202]),
    ]

    assert decode_tokens == 0
    assert prefill_tokens == 256

def test_step_empty_schedule(make_engine):
    """没有可执行请求时，不应该调用 ModelRunner。"""

    schedule_output = ScheduleOutput(
        decode_seqs=[],
        prefill_seqs=[],
        decode_tokens=0,
        prefill_tokens=0,
    )

    engine, events = make_engine(schedule_output)

    outputs, decode_tokens, prefill_tokens = engine.step()

    assert events == ["schedule"]

    assert outputs == []
    assert decode_tokens == 0
    assert prefill_tokens == 0
