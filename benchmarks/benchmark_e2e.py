import sys
sys.path.insert(0, "/home/chen17/nano-vllm")
import os
import statistics
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams


# ==================== 测试配置 ====================

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B")

NUM_REQUESTS = 4
PROMPT_TOKENS = 128
MAX_NEW_TOKENS = 32

WARMUP = 1
REPEATS = 5

MAX_NUM_SEQS = 4
MAX_BATCHED_TOKENS = 16384


def make_prompts(tokenizer):
    """构造固定长度的输入 Token，保证每次实验输入一致。"""

    texts = [
        "Explain how GPU parallel computing works.",
        "Explain the architecture of a large language model.",
        "Describe the principles of memory optimization.",
        "Explain how an inference engine works.",
    ]

    prompts = []

    for text in texts[:NUM_REQUESTS]:
        ids = tokenizer.encode(text, add_special_tokens=False)

        # 重复 Token，构造固定长度输入
        while len(ids) < PROMPT_TOKENS:
            ids += ids

        prompts.append(ids[:PROMPT_TOKENS])

    return prompts


def run_one_batch(llm, prompts):
    """运行一个 Batch，并记录每条请求的 Token 时间。"""

    sampling_params = [
        SamplingParams(
            temperature=0.6,
            max_tokens=MAX_NEW_TOKENS,
            ignore_eos=True,
        )
        for _ in prompts
    ]

    start = perf_counter()

    # 提交请求
    for prompt, params in zip(prompts, sampling_params):
        llm.add_request(prompt, params)

    # 保存 Sequence 引用，后续即使请求完成并离开队列，
    # 仍然可以读取它的生成 Token 数量
    seqs = list(llm.scheduler.waiting)

    last_counts = {seq.seq_id: 0 for seq in seqs}
    token_times = {seq.seq_id: [] for seq in seqs}

    # 逐 Step 执行推理
    while not llm.is_finished():
        llm.step()

        now = perf_counter()

        for seq in seqs:
            seq_id = seq.seq_id
            current_count = seq.num_completion_tokens
            previous_count = last_counts[seq_id]

            # 发现新生成的 Token
            new_tokens = current_count - previous_count

            if new_tokens > 0:
                token_times[seq_id].extend([now] * new_tokens)
                last_counts[seq_id] = current_count

    torch.cuda.synchronize()
    end = perf_counter()

    # 计算每条请求的指标
    ttfts = []
    tpot_list = []
    completion_times = []

    total_tokens = 0

    for seq in seqs:
        times = token_times[seq.seq_id]

        if len(times) != MAX_NEW_TOKENS:
            raise RuntimeError(
                f"Request {seq.seq_id}: "
                f"expected {MAX_NEW_TOKENS} tokens, "
                f"got {len(times)}"
            )

        ttfts.append((times[0] - start) * 1000)

        if len(times) > 1:
            tpot = (times[-1] - times[0]) / (len(times) - 1)
            tpot_list.append(tpot * 1000)

        completion_times.append((times[-1] - start) * 1000)
        total_tokens += len(times)

    wall_time = end - start

    return {
        "ttft_ms": statistics.mean(ttfts),
        "tpot_ms": statistics.mean(tpot_list),
        "completion_ms": statistics.mean(completion_times),
        "throughput": total_tokens / wall_time,
        "wall_time_ms": wall_time * 1000,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    use_triton = os.getenv("NANOVLLM_USE_TRITON_SILU", "1") == "1"

    print("=" * 60)
    print("nano-vLLM End-to-End Benchmark")
    print("=" * 60)

    print("GPU:", torch.cuda.get_device_name())
    print("Model:", MODEL)
    print("SwiGLU:", "Triton" if use_triton else "PyTorch")
    print("Requests:", NUM_REQUESTS)
    print("Prompt tokens:", PROMPT_TOKENS)
    print("Output tokens:", MAX_NEW_TOKENS)

    # 创建引擎
    llm = LLM(
        MODEL,
        enforce_eager=True,
        max_model_len=4096,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_BATCHED_TOKENS,
    )

    prompts = make_prompts(llm.tokenizer)

    print("\n========== Warmup ==========")

    for _ in range(WARMUP):
        params = [
            SamplingParams(
                temperature=0.6,
                max_tokens=4,
                ignore_eos=True,
            )
            for _ in prompts
        ]

        llm.generate(prompts, params, use_tqdm=False)

    torch.cuda.synchronize()

    # 重置峰值统计，但保留模型本身的显存占用
    torch.cuda.reset_peak_memory_stats()

    print("\n========== Benchmark ==========")

    results = []

    for i in range(REPEATS):
        torch.manual_seed(0)

        result = run_one_batch(llm, prompts)
        results.append(result)

        print(
            f"Run {i + 1}/{REPEATS} | "
            f"TTFT={result['ttft_ms']:.3f} ms | "
            f"TPOT={result['tpot_ms']:.3f} ms | "
            f"Throughput={result['throughput']:.2f} tok/s | "
            f"Completion={result['completion_ms']:.3f} ms"
        )

    # 统计平均值
    print("\n========== Average Results ==========")

    for key, label in [
        ("ttft_ms", "TTFT"),
        ("tpot_ms", "TPOT"),
        ("throughput", "Output Tokens/s"),
        ("completion_ms", "Request Completion"),
        ("wall_time_ms", "Batch Wall Time"),
    ]:
        values = [r[key] for r in results]
        print(f"{label:22s}: {statistics.mean(values):.3f}")

    peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)

    print(f"\nPeak VRAM Allocated : {peak_allocated:.3f} GiB")
    print(f"Peak VRAM Reserved  : {peak_reserved:.3f} GiB")


if __name__ == "__main__":
    main()
