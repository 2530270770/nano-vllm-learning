
import os

import pytest
import torch

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import get_context


MODEL_PATH = os.getenv(
    "NANOVLLM_MODEL_PATH",
    "/home/chen17/huggingface/Qwen3-0.6B",
)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or os.getenv("RUN_MODEL_RUNNER_GPU_TESTS") != "1",
    reason="需要显式启用真实 GPU 集成测试",
)
def test_model_runner_prefill_then_decode():
    assert os.path.isdir(MODEL_PATH), (
        f"模型目录不存在: {MODEL_PATH}"
    )

    # 1. 创建轻量测试配置
    config = Config(
        model=MODEL_PATH,
        max_num_batched_tokens=256,
        max_num_seqs=1,
        max_num_decode_seqs=1,
        max_model_len=256,
        gpu_memory_utilization=0.70,
        tensor_parallel_size=1,
        enforce_eager=True,
    )

    Sequence.block_size = config.kvcache_block_size

    # 2. 初始化真实 ModelRunner
    # event=[] 表示单卡，不需要 TP 通信事件
    runner = ModelRunner(config, rank=0, event=[])

    try:
        scheduler = Scheduler(config)

        # 3. 构造一个真实请求
        seq = Sequence(
            token_ids=list(range(1, 33)),
            sampling_params=SamplingParams(
                temperature=0.6,
                max_tokens=2,
                ignore_eos=True,
            ),
        )

        scheduler.add(seq)

        # ==========================================
        # 4. 测试 Prefill
        # ==========================================

        prefill_output = scheduler.schedule()

        assert prefill_output.prefill_seqs == [seq]
        assert prefill_output.decode_seqs == []

        prefill_token_ids = runner.run_prefill(
            prefill_output.prefill_seqs
        )

        assert len(prefill_token_ids) == 1
        assert isinstance(prefill_token_ids[0], int)

        scheduler.postprocess(
            prefill_output.prefill_seqs,
            prefill_token_ids,
            is_prefill=True,
        )

        assert seq.status == SequenceStatus.RUNNING
        assert seq.num_cached_tokens == 32
        assert seq.num_completion_tokens == 1

        # ==========================================
        # 5. 测试 Decode
        # ==========================================

        decode_output = scheduler.schedule()

        assert decode_output.decode_seqs == [seq]
        assert decode_output.prefill_seqs == []

        decode_token_ids = runner.run_decode(
            decode_output.decode_seqs
        )

        assert len(decode_token_ids) == 1
        assert isinstance(decode_token_ids[0], int)

        scheduler.postprocess(
            decode_output.decode_seqs,
            decode_token_ids,
            is_prefill=False,
        )

        # ==========================================
        # 6. 检查最终状态
        # ==========================================

        assert seq.is_finished
        assert seq.num_completion_tokens == 2
        assert scheduler.is_finished()

        # 两条执行路径结束后，全局 Context 都应清理
        context = get_context()
        assert context.is_prefill is False
        assert context.slot_mapping is None
        assert context.context_lens is None

        torch.cuda.synchronize()

        print("\nModelRunner GPU integration test passed")
        print(f"Prefill token: {prefill_token_ids[0]}")
        print(f"Decode token: {decode_token_ids[0]}")
        print(f"Completion tokens: {seq.completion_token_ids}")
        print(
            f"GPU memory allocated: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GiB"
        )

    finally:
        runner.exit()
