
import os

import pytest
import torch

from nanovllm.config import Config
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


MODEL_PATH = os.getenv(
    "NANOVLLM_MODEL_PATH",
    "/home/chen17/huggingface/Qwen3-0.6B",
)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or os.getenv("RUN_MODEL_RUNNER_GPU_TESTS") != "1",
    reason="需要显式启用真实 GPU 集成测试",
)
def test_mixed_batch_decode_and_chunked_prefill():
    assert os.path.isdir(MODEL_PATH), f"模型目录不存在: {MODEL_PATH}"

    # 1. 创建测试配置
    config = Config(
        model=MODEL_PATH,
        max_num_batched_tokens=64,
        max_num_seqs=4,
        max_num_decode_seqs=1,
        max_model_len=256,
        gpu_memory_utilization=0.70,
        tensor_parallel_size=1,
        enforce_eager=True,
    )

    Sequence.block_size = config.kvcache_block_size

    # 2. 初始化真实 ModelRunner 和 Scheduler
    runner = ModelRunner(config, rank=0, event=[])

    try:
        engine = LLMEngine.__new__(LLMEngine)
        engine.model_runner = runner
        engine.scheduler = Scheduler(config)

        # 记录真实执行顺序，但不替换真实模型计算
        execution_order = []
        original_call = runner.call

        def tracked_call(method_name, *args):
            execution_order.append(method_name)
            return original_call(method_name, *args)

        runner.call = tracked_call

        # 3. 创建一个短请求，先完成 Prefill，进入 Decode
        decode_seq = Sequence(
            list(range(1, 17)),
            SamplingParams(
                temperature=0.6,
                max_tokens=4,
                ignore_eos=True,
            ),
        )

        engine.scheduler.add(decode_seq)

        # 第一次 Engine Step：只有 Prefill
        outputs, decode_tokens, prefill_tokens = engine.step()

        assert decode_tokens == 0
        assert prefill_tokens == 16
        assert decode_seq.status == SequenceStatus.RUNNING
        assert decode_seq.num_cached_tokens == 16

        # 4. 新增两个较长请求，等待 Chunked Prefill
        prefill_seq_1 = Sequence(
            list(range(100, 196)),
            SamplingParams(
                temperature=0.6,
                max_tokens=4,
                ignore_eos=True,
            ),
        )

        prefill_seq_2 = Sequence(
            list(range(200, 296)),
            SamplingParams(
                temperature=0.6,
                max_tokens=4,
                ignore_eos=True,
            ),
        )

        engine.scheduler.add(prefill_seq_1)
        engine.scheduler.add(prefill_seq_2)

        execution_order.clear()

        # 5. 第二次 Engine Step：Decode + Chunked Prefill
        outputs, decode_tokens, prefill_tokens = engine.step()

        torch.cuda.synchronize()

        # 验证同一个 Step 确实执行了两个阶段
        assert execution_order == [
            "run_decode",
            "run_prefill",
        ]

        # Decode 只调度一个请求，占用一个 Token
        assert decode_tokens == 1

        # 剩余 63 个 Token 分配给两个 Prefill 请求
        assert prefill_tokens == 63

        # 验证两个 Prefill 请求都获得了 Chunk
        assert prefill_seq_1.num_cached_tokens == 32
        assert prefill_seq_2.num_cached_tokens == 31

        # 两个 Prefill 请求都还没有完成 Prefill
        assert prefill_seq_1.status == SequenceStatus.WAITING
        assert prefill_seq_2.status == SequenceStatus.WAITING

        # Decode 请求已经执行一次 Decode
        assert decode_seq.num_completion_tokens == 2
        assert decode_seq.status == SequenceStatus.RUNNING

        # Prefill 的 Chunk 没有错误地追加生成 Token
        assert prefill_seq_1.num_completion_tokens == 0
        assert prefill_seq_2.num_completion_tokens == 0

        # 所有请求的本轮调度 Token 数量都已清零
        assert decode_seq.num_scheduled_tokens == 0
        assert prefill_seq_1.num_scheduled_tokens == 0
        assert prefill_seq_2.num_scheduled_tokens == 0

        # KV Cache Block 已经正确分配
        assert len(decode_seq.block_table) > 0
        assert len(prefill_seq_1.block_table) > 0
        assert len(prefill_seq_2.block_table) > 0

        # 6. 连续执行后续 Engine Step，验证 Chunk 能持续推进
        expected_progress = [
            (64, 62, 63),
            (96, 93, 63),
            (97, 96, 3),
        ]

        for expected_1, expected_2, expected_prefill_tokens in expected_progress:
            execution_order.clear()

            outputs, decode_tokens, prefill_tokens = engine.step()

            torch.cuda.synchronize()

            assert execution_order == [
                "run_decode",
                "run_prefill",
            ]

            assert decode_tokens == 1
            assert prefill_tokens == expected_prefill_tokens

            assert prefill_seq_1.num_cached_tokens == expected_1
            assert prefill_seq_2.num_cached_tokens == expected_2

        # 两个 Prefill 请求最终都完成 Prefill，进入 Running
        assert prefill_seq_1.status == SequenceStatus.RUNNING
        assert prefill_seq_2.status == SequenceStatus.RUNNING

        # 完成 Prefill 后，各自追加了一个生成 Token, 因为请求 1 比请求 2 早完成 Prefill，所以它多经历了一轮 Decode。
        assert prefill_seq_1.num_completion_tokens == 2
        assert prefill_seq_2.num_completion_tokens == 1

        print("Multi-step Chunked Prefill passed")
        print(
            "Final cached tokens:",
            prefill_seq_1.num_cached_tokens,
            prefill_seq_2.num_cached_tokens,
        )
        print(
            "Final completion tokens:",
            prefill_seq_1.num_completion_tokens,
            prefill_seq_2.num_completion_tokens,
        )


        print("\nMixed-Batch GPU integration test passed")
        print(f"Execution order: {execution_order}")
        print(f"Decode tokens: {decode_tokens}")
        print(f"Prefill tokens: {prefill_tokens}")
        print(
            "Prefill cached tokens:",
            prefill_seq_1.num_cached_tokens,
            prefill_seq_2.num_cached_tokens,
        )
        print(
            "GPU memory allocated:",
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GiB",
        )

    finally:
        runner.exit()
