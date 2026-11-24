
import torch
import pytest

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.utils.context import get_context, set_context


class FakeSampler:
    def __call__(self, logits, temperatures):
        return torch.tensor([101, 202])


@pytest.fixture
def runner(monkeypatch):
    """创建不初始化 CUDA/NCCL 的 ModelRunner。"""
    model_runner = ModelRunner.__new__(ModelRunner)
    model_runner.rank = 0
    model_runner.sampler = FakeSampler()

    calls = []

    monkeypatch.setattr(
        model_runner,
        "prepare_prefill",
        lambda seqs: (
            calls.append(("prepare_prefill", seqs))
            or (torch.tensor([1, 2]), torch.tensor([0, 1]))
        ),
    )

    monkeypatch.setattr(
        model_runner,
        "prepare_decode",
        lambda seqs: (
            calls.append(("prepare_decode", seqs))
            or (torch.tensor([3]), torch.tensor([2]))
        ),
    )

    monkeypatch.setattr(
        model_runner,
        "prepare_sample",
        lambda seqs: torch.tensor([1.0] * len(seqs)),
    )

    monkeypatch.setattr(
        model_runner,
        "run_model",
        lambda input_ids, positions, is_prefill: (
            calls.append(("run_model", is_prefill))
            or torch.zeros(len(input_ids), 10)
        ),
    )

    model_runner._test_calls = calls
    return model_runner


def test_run_prefill(runner):
    """Prefill 应走 prepare_prefill 和 Eager 执行路径。"""
    seqs = [object(), object()]

    token_ids = runner.run_prefill(seqs)

    assert token_ids == [101, 202]
    assert runner._test_calls == [
        ("prepare_prefill", seqs),
        ("run_model", True),
    ]


def test_run_decode(runner):
    """Decode 应走 prepare_decode 和 Decode 执行路径。"""
    seqs = [object()]

    token_ids = runner.run_decode(seqs)

    assert token_ids == [101, 202]
    assert runner._test_calls == [
        ("prepare_decode", seqs),
        ("run_model", False),
    ]


def test_run_prefill_resets_context(runner):
    """Prefill 执行结束后应清理全局 Context。"""
    set_context(True, max_seqlen_q=10)

    runner.run_prefill([object(), object()])

    context = get_context()
    assert context.is_prefill is False
    assert context.max_seqlen_q == 0


def test_run_decode_resets_context(runner):
    """Decode 执行结束后应清理全局 Context。"""
    set_context(False, context_lens=torch.tensor([10]))

    runner.run_decode([object()])

    context = get_context()
    assert context.context_lens is None
    assert context.slot_mapping is None
