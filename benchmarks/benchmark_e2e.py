import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import csv
import statistics
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams


# ============================================================
# Benchmark 配置
# ============================================================

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B")

NUM_REQUESTS = 4
PROMPT_TOKENS = 128
MAX_NEW_TOKENS = 32

WARMUP = 1
REPEATS = 5

MAX_NUM_SEQS = 4
MAX_BATCHED_TOKENS = 16384

OUTPUT_DIR = Path("outputs/benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Prompt
# ============================================================

def make_prompts(tokenizer):
    texts = [
        "Explain how GPU parallel computing works.",
        "Explain the architecture of a large language model.",
        "Describe the principles of memory optimization.",
        "Explain how an inference engine works.",
    ]

    prompts = []

    for text in texts[:NUM_REQUESTS]:
        ids = tokenizer.encode(text, add_special_tokens=False)

        while len(ids) < PROMPT_TOKENS:
            ids += ids

        prompts.append(ids[:PROMPT_TOKENS])

    return prompts


# ============================================================
# 单轮 Batch 测试
# ============================================================

def run_one_batch(llm, prompts):
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            max_tokens=MAX_NEW_TOKENS,
            ignore_eos=True,
        )
        for _ in prompts
    ]

    start = perf_counter()

    for prompt, params in zip(prompts, sampling_params):
        llm.add_request(prompt, params)

    # 保存本轮所有 Sequence
    seqs = list(llm.scheduler.waiting)

    last_counts = {
        seq.seq_id: 0 for seq in seqs
    }

    token_times = {
        seq.seq_id: [] for seq in seqs
    }

    # 持续执行 Scheduler Step
    while not llm.is_finished():
        llm.step()

        now = perf_counter()

        for seq in seqs:
            seq_id = seq.seq_id

            current_count = seq.num_completion_tokens
            previous_count = last_counts[seq_id]

            new_tokens = current_count - previous_count

            if new_tokens > 0:
                token_times[seq_id].extend(
                    [now] * new_tokens
                )
                last_counts[seq_id] = current_count

    torch.cuda.synchronize()
    end = perf_counter()

    # ========================================================
    # 计算指标
    # ========================================================

    ttfts = []
    tpots = []
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

        ttft = (times[0] - start) * 1000
        ttfts.append(ttft)

        if len(times) > 1:
            tpot = (
                (times[-1] - times[0])
                / (len(times) - 1)
            ) * 1000

            tpots.append(tpot)

        completion = (times[-1] - start) * 1000
        completion_times.append(completion)

        total_tokens += len(times)

    wall_time = end - start

    return {
        "ttft_ms": statistics.mean(ttfts),
        "tpot_ms": statistics.mean(tpots),
        "completion_ms": statistics.mean(completion_times),
        "throughput": total_tokens / wall_time,
        "wall_time_ms": wall_time * 1000,
    }


# ============================================================
# 统计
# ============================================================

def summarize(results):
    summary = {}

    for key in results[0]:
        values = [item[key] for item in results]

        summary[key] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": (
                statistics.stdev(values)
                if len(values) > 1 else 0.0
            ),
        }

    return summary


# ============================================================
# 主函数
# ============================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    impl = os.getenv("NANOVLLM_SILU_IMPL", "triton")

    if impl not in ("eager", "compile", "triton"):
        raise ValueError(
            "NANOVLLM_SILU_IMPL must be "
            "eager, compile, or triton"
        )

    print("=" * 65)
    print("nano-vLLM End-to-End Benchmark")
    print("=" * 65)

    print("GPU:", torch.cuda.get_device_name())
    print("PyTorch:", torch.__version__)
    print("Model:", MODEL)
    print("SwiGLU implementation:", impl)
    print("Requests:", NUM_REQUESTS)
    print("Prompt tokens:", PROMPT_TOKENS)
    print("Output tokens:", MAX_NEW_TOKENS)
    print("Warmup:", WARMUP)
    print("Repeats:", REPEATS)

    # 创建引擎
    llm = LLM(
        MODEL,
        enforce_eager=True,
        max_model_len=4096,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_BATCHED_TOKENS,
    )

    prompts = make_prompts(llm.tokenizer)

    # ========================================================
    # Warmup
    # ========================================================

    print("\n========== Warmup ==========")

    for i in range(WARMUP):
        params = [
            SamplingParams(
                temperature=0.6,
                max_tokens=4,
                ignore_eos=True,
            )
            for _ in prompts
        ]

        llm.generate(
            prompts,
            params,
            use_tqdm=False,
        )

        print(f"Warmup {i + 1}/{WARMUP} completed")

    torch.cuda.synchronize()

    # 不把模型初始化和 Warmup 之前的峰值计入结果
    torch.cuda.reset_peak_memory_stats()

    # ========================================================
    # Benchmark
    # ========================================================

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
            f"Completion={result['completion_ms']:.3f} ms | "
            f"Wall={result['wall_time_ms']:.3f} ms"
        )

    # ========================================================
    # Summary
    # ========================================================

    summary = summarize(results)

    print("\n========== Summary ==========")

    labels = {
        "ttft_ms": "TTFT (ms)",
        "tpot_ms": "TPOT (ms)",
        "throughput": "Output Throughput (tok/s)",
        "completion_ms": "Completion (ms)",
        "wall_time_ms": "Batch Wall Time (ms)",
    }

    for key, label in labels.items():
        item = summary[key]

        print(
            f"{label:28s} "
            f"mean={item['mean']:.3f}, "
            f"median={item['median']:.3f}, "
            f"std={item['stdev']:.3f}"
        )

    peak_allocated = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved() / (1024 ** 3)
    )

    print(f"\nPeak VRAM Allocated: {peak_allocated:.3f} GiB")
    print(f"Peak VRAM Reserved : {peak_reserved:.3f} GiB")

    # ========================================================
    # CSV
    # ========================================================

    csv_path = OUTPUT_DIR / f"e2e_{impl}.csv"

    fieldnames = [
        "implementation",
        "run",
        "ttft_ms",
        "tpot_ms",
        "completion_ms",
        "throughput",
        "wall_time_ms",
        "peak_allocated_gib",
        "peak_reserved_gib",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for i, result in enumerate(results, start=1):
            writer.writerow({
                "implementation": impl,
                "run": i,
                **result,
                "peak_allocated_gib": peak_allocated,
                "peak_reserved_gib": peak_reserved,
            })

    print(f"\nResults saved to: {csv_path}")

    # 显式释放引擎资源
    # llm.exit()


if __name__ == "__main__":
    main()
