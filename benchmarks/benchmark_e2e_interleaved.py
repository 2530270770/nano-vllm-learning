import os
import sys
import csv
import math
import random
import statistics
from pathlib import Path
from time import perf_counter
from datetime import datetime

# ============================================================
# 路径配置
# ============================================================

# 自动定位 nano-vLLM 项目根目录，不依赖固定的 /home/chen17 路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from nanovllm import LLM, SamplingParams


# ============================================================
# Benchmark 配置
# ============================================================

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B")

NUM_REQUESTS = 4
PROMPT_TOKENS = 128
MAX_NEW_TOKENS = 32

# 每种实现预热次数
WARMUP = 1

# 正式测试轮数：每种实现各测试 15 次
REPEATS = 15

MAX_NUM_SEQS = 4
MAX_BATCHED_TOKENS = 16384

# 三种 SwiGLU 实现
IMPLEMENTATIONS = [
    "eager",
    "compile",
    "triton",
]

# 固定随机种子，保证每次运行的测试顺序可复现
ORDER_SEED = 2026
TORCH_SEED = 0

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "benchmark"

RAW_CSV = OUTPUT_DIR / "e2e_interleaved_raw.csv"
SUMMARY_CSV = OUTPUT_DIR / "e2e_interleaved_summary.csv"


# ============================================================
# Prompt 构造
# ============================================================

def make_prompts(tokenizer):
    """
    构造固定长度的输入 Token。

    每次测试使用完全相同的 Prompt，
    避免输入长度差异影响测试结果。
    """

    texts = [
        "Explain how GPU parallel computing works.",
        "Explain the architecture of a large language model.",
        "Describe the principles of memory optimization.",
        "Explain how an inference engine works.",
    ]

    prompts = []

    for text in texts[:NUM_REQUESTS]:
        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        # 重复 Token，构造固定长度输入
        while len(ids) < PROMPT_TOKENS:
            ids += ids

        prompts.append(ids[:PROMPT_TOKENS])

    return prompts


# ============================================================
# 设置 SwiGLU 实现
# ============================================================

def set_silu_impl(impl):
    """
    动态切换 SwiGLU 实现。

    要求 activation.py 的 forward() 每次调用时
    都读取 NANOVLLM_SILU_IMPL。
    """

    if impl not in IMPLEMENTATIONS:
        raise ValueError(
            f"Unsupported implementation: {impl}"
        )

    os.environ["NANOVLLM_SILU_IMPL"] = impl

    print(
        f"\n[SwiGLU] Switched implementation to: "
        f"{impl.upper()}"
    )


# ============================================================
# RNG 设置
# ============================================================

def reset_random_seed():
    """
    每次测试使用相同随机种子。

    让三种实现尽可能使用相同的采样随机数序列，
    减少生成过程随机性的影响。
    """

    torch.manual_seed(TORCH_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TORCH_SEED)


# ============================================================
# 执行一个 Batch
# ============================================================

def run_one_batch(llm, prompts):
    """
    执行一个完整 Batch。

    记录：
    - TTFT
    - TPOT
    - Completion Time
    - Output Throughput
    - Batch Wall Time

    注意：
    这里的时间包含 Python Scheduler Loop，
    并不是纯 GPU Kernel 延迟。
    """

    sampling_params = [
        SamplingParams(
            temperature=0.6,
            max_tokens=MAX_NEW_TOKENS,
            ignore_eos=True,
        )
        for _ in prompts
    ]

    # 确保上一次 GPU 工作已经完成
    torch.cuda.synchronize()

    start = perf_counter()

    # 提交请求
    for prompt, params in zip(prompts, sampling_params):
        llm.add_request(prompt, params)

    # 保存 Sequence 引用
    seqs = list(llm.scheduler.waiting)

    last_counts = {
        seq.seq_id: 0
        for seq in seqs
    }

    token_times = {
        seq.seq_id: []
        for seq in seqs
    }

    # 逐 Step 执行推理
    while not llm.is_finished():

        llm.step()

        now = perf_counter()

        for seq in seqs:

            seq_id = seq.seq_id

            current_count = seq.num_completion_tokens
            previous_count = last_counts[seq_id]

            # 当前 Step 新生成的 Token 数量
            new_tokens = current_count - previous_count

            if new_tokens > 0:

                token_times[seq_id].extend(
                    [now] * new_tokens
                )

                last_counts[seq_id] = current_count

    # 等待 GPU 工作完成
    torch.cuda.synchronize()

    end = perf_counter()

    # ========================================================
    # 计算指标
    # ========================================================

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

        # TTFT
        ttft = times[0] - start

        ttfts.append(
            ttft * 1000
        )

        # TPOT
        if len(times) > 1:

            tpot = (
                times[-1] - times[0]
            ) / (len(times) - 1)

            tpot_list.append(
                tpot * 1000
            )

        # 单请求完成时间
        completion_time = times[-1] - start

        completion_times.append(
            completion_time * 1000
        )

        total_tokens += len(times)

    wall_time = end - start

    throughput = total_tokens / wall_time

    return {
        "ttft_ms": statistics.mean(ttfts),
        "tpot_ms": statistics.mean(tpot_list),
        "completion_ms": statistics.mean(completion_times),
        "throughput": throughput,
        "wall_time_ms": wall_time * 1000,
    }


# ============================================================
# 单次测试
# ============================================================

def benchmark_once(llm, prompts, impl, round_id, order_id):
    """
    执行某种实现的一次正式测试。
    """

    set_silu_impl(impl)

    reset_random_seed()

    # 每次测试前重置 CUDA 峰值显存统计
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    result = run_one_batch(
        llm,
        prompts,
    )

    peak_allocated = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved()
        / (1024 ** 3)
    )

    result["peak_allocated_gib"] = peak_allocated
    result["peak_reserved_gib"] = peak_reserved

    result["round"] = round_id
    result["order"] = order_id
    result["implementation"] = impl

    print(
        f"Round {round_id:02d}/{REPEATS} | "
        f"Order {order_id} | "
        f"{impl.upper():7s} | "
        f"TTFT={result['ttft_ms']:.3f} ms | "
        f"TPOT={result['tpot_ms']:.3f} ms | "
        f"Throughput={result['throughput']:.2f} tok/s | "
        f"Completion={result['completion_ms']:.3f} ms | "
        f"Allocated={peak_allocated:.3f} GiB"
    )

    return result


# ============================================================
# 预热
# ============================================================

def warmup_one_impl(llm, prompts, impl):
    """
    对某种实现进行完整预热。

    使用与正式测试相同的：
    - Prompt 长度
    - Batch Size
    - Output Token 数量

    这样 Compile 可以提前完成对应 Shape 的编译。
    """

    set_silu_impl(impl)

    print(
        f"[Warmup] {impl.upper()} "
        f"{WARMUP} time(s)"
    )

    for i in range(WARMUP):

        reset_random_seed()

        sampling_params = [
            SamplingParams(
                temperature=0.6,
                max_tokens=MAX_NEW_TOKENS,
                ignore_eos=True,
            )
            for _ in prompts
        ]

        llm.generate(
            prompts,
            sampling_params,
            use_tqdm=False,
        )

        torch.cuda.synchronize()

        print(
            f"[Warmup] {impl.upper()} "
            f"{i + 1}/{WARMUP} completed"
        )


# ============================================================
# P95 计算
# ============================================================

def percentile(values, percent):
    """
    使用 nearest-rank 方法计算百分位数。

    例如 15 个样本的 P95 为排序后的第 15 个值。
    """

    sorted_values = sorted(values)

    if not sorted_values:
        return float("nan")

    rank = math.ceil(
        percent / 100 * len(sorted_values)
    )

    rank = max(
        1,
        min(rank, len(sorted_values)),
    )

    return sorted_values[rank - 1]


# ============================================================
# 汇总统计
# ============================================================

def calculate_summary(results):
    """
    按实现分别计算：

    Mean
    Median
    P95
    Std
    """

    metrics = [
        "ttft_ms",
        "tpot_ms",
        "throughput",
        "completion_ms",
        "wall_time_ms",
        "peak_allocated_gib",
        "peak_reserved_gib",
    ]

    summary = []

    for impl in IMPLEMENTATIONS:

        impl_results = [
            item
            for item in results
            if item["implementation"] == impl
        ]

        for metric in metrics:

            values = [
                item[metric]
                for item in impl_results
            ]

            row = {
                "implementation": impl,
                "metric": metric,
                "count": len(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p95": percentile(values, 95),
                "std": (
                    statistics.stdev(values)
                    if len(values) > 1
                    else 0.0
                ),
            }

            summary.append(row)

    return summary


# ============================================================
# CSV 保存
# ============================================================

def save_raw_csv(results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "round",
        "order",
        "implementation",
        "ttft_ms",
        "tpot_ms",
        "throughput",
        "completion_ms",
        "wall_time_ms",
        "peak_allocated_gib",
        "peak_reserved_gib",
    ]

    with open(
        RAW_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result)


def save_summary_csv(summary):

    fieldnames = [
        "implementation",
        "metric",
        "count",
        "mean",
        "median",
        "p95",
        "std",
    ]

    with open(
        SUMMARY_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in summary:
            writer.writerow(row)


# ============================================================
# 打印最终汇总
# ============================================================

def print_summary(summary):

    metric_labels = {
        "ttft_ms": "TTFT (ms)",
        "tpot_ms": "TPOT (ms)",
        "throughput": "Throughput (tok/s)",
        "completion_ms": "Completion (ms)",
        "wall_time_ms": "Wall Time (ms)",
        "peak_allocated_gib": "Peak Allocated (GiB)",
        "peak_reserved_gib": "Peak Reserved (GiB)",
    }

    print("\n")
    print("=" * 100)
    print("E2E INTERLEAVED BENCHMARK SUMMARY")
    print("=" * 100)

    for metric, label in metric_labels.items():

        print(f"\n{label}")

        print(
            f"{'Implementation':<15}"
            f"{'Mean':>12}"
            f"{'Median':>12}"
            f"{'P95':>12}"
            f"{'Std':>12}"
        )

        print("-" * 63)

        for impl in IMPLEMENTATIONS:

            row = next(
                item
                for item in summary
                if item["implementation"] == impl
                and item["metric"] == metric
            )

            print(
                f"{impl:<15}"
                f"{row['mean']:>12.3f}"
                f"{row['median']:>12.3f}"
                f"{row['p95']:>12.3f}"
                f"{row['std']:>12.3f}"
            )


# ============================================================
# 主函数
# ============================================================

def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("nano-vLLM E2E Interleaved Benchmark")
    print("=" * 70)

    print("Project:", PROJECT_ROOT)
    print("GPU:", torch.cuda.get_device_name())
    print("PyTorch:", torch.__version__)
    print("Model:", MODEL)

    print("Implementations:", IMPLEMENTATIONS)

    print("Requests:", NUM_REQUESTS)
    print("Prompt tokens:", PROMPT_TOKENS)
    print("Output tokens:", MAX_NEW_TOKENS)

    print("Warmup:", WARMUP)
    print("Repeats:", REPEATS)

    print("Tensor Parallel Size: 1")

    # ========================================================
    # 初始化 LLMEngine
    # ========================================================

    print("\n========== Initialize LLM ==========")

    # 先设置初始实现
    set_silu_impl("eager")

    llm = LLM(
        MODEL,
        enforce_eager=True,
        max_model_len=4096,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_BATCHED_TOKENS,
        tensor_parallel_size=1,
    )

    prompts = make_prompts(
        llm.tokenizer
    )

    torch.cuda.synchronize()

    print("LLM initialization completed.")

    # ========================================================
    # Warmup
    # ========================================================

    print("\n========== Warmup All Implementations ==========")

    warmup_rng = random.Random(
        ORDER_SEED
    )

    warmup_order = IMPLEMENTATIONS.copy()

    warmup_rng.shuffle(
        warmup_order
    )

    print(
        "Warmup order:",
        " -> ".join(warmup_order),
    )

    for impl in warmup_order:

        warmup_one_impl(
            llm,
            prompts,
            impl,
        )

    torch.cuda.synchronize()

    # ========================================================
    # 正式测试
    # ========================================================

    print("\n========== Interleaved Benchmark ==========")

    rng = random.Random(
        ORDER_SEED
    )

    results = []

    for round_id in range(1, REPEATS + 1):

        print("\n" + "=" * 70)
        print(
            f"Round {round_id}/{REPEATS}"
        )
        print("=" * 70)

        # 每一轮随机打乱三种实现的执行顺序
        round_order = IMPLEMENTATIONS.copy()

        rng.shuffle(
            round_order
        )

        print(
            "Execution order:",
            " -> ".join(round_order),
        )

        for order_id, impl in enumerate(
            round_order,
            start=1,
        ):

            result = benchmark_once(
                llm=llm,
                prompts=prompts,
                impl=impl,
                round_id=round_id,
                order_id=order_id,
            )

            results.append(result)

    # ========================================================
    # 保存结果
    # ========================================================

    print("\n========== Save Results ==========")

    summary = calculate_summary(
        results
    )

    save_raw_csv(
        results
    )

    save_summary_csv(
        summary
    )

    # ========================================================
    # 输出统计结果
    # ========================================================

    print_summary(
        summary
    )

    print("\n========== CSV Files ==========")

    print("Raw results:")
    print(RAW_CSV)

    print("\nSummary:")
    print(SUMMARY_CSV)

    print("\nBenchmark completed.")


if __name__ == "__main__":
    main()