
import sys
import csv
import math
import random
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.testing


# ============================================================
# 1. 项目路径
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nanovllm.layers.triton_kernels import _silu_and_mul_kernel


# ============================================================
# 2. Benchmark 配置
# ============================================================

SHAPES = [
    (512, 6144),  # Prefill
    (4, 6144),    # Decode
]

DTYPE = torch.bfloat16

# Prefill 候选配置
PREFILL_CONFIGS = [
    (256, 4),
    (512, 4),
    (512, 8),
    (1024, 4),
    (2048, 2),
]

# Decode 候选配置
DECODE_CONFIGS = [
    (256, 2),
    (512, 4),
    (512, 8),
]

WARMUP = 10
ROUNDS = 30
CUDA_GRAPH_REP = 100

RANDOM_SEED = 2026

OUTPUT_DIR = REPO_ROOT / "outputs" / "benchmark"

RAW_CSV_PATH = OUTPUT_DIR / "swiglu_stability_raw.csv"
SUMMARY_CSV_PATH = OUTPUT_DIR / "swiglu_stability_summary.csv"


# ============================================================
# 3. PyTorch 实现
# ============================================================

def swiglu_torch(x):
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


swiglu_compile = torch.compile(swiglu_torch)


# ============================================================
# 4. Triton Kernel 调用接口
# ============================================================

def swiglu_triton_config(x, block_size, num_warps):

    if not x.is_cuda:
        raise ValueError("Input must be CUDA tensor")

    if not x.is_contiguous():
        raise ValueError("Input must be contiguous")

    if x.shape[-1] % 2 != 0:
        raise ValueError("Last dimension must be divisible by 2")

    n = x.shape[-1] // 2
    x_2d = x.view(-1, 2 * n)

    m = x_2d.shape[0]

    output = torch.empty(
        (*x.shape[:-1], n),
        device=x.device,
        dtype=x.dtype,
    )

    grid = (triton.cdiv(m * n, block_size),)

    _silu_and_mul_kernel[grid](
        x_2d,
        output,
        m,
        n,
        block_size,
        num_warps=num_warps,
    )

    return output


# ============================================================
# 5. 正确性检查
# ============================================================

def check_correctness(y_reference, y_test):

    try:
        torch.testing.assert_close(
            y_test,
            y_reference,
            rtol=2e-2,
            atol=2e-2,
        )
        return True

    except AssertionError as e:
        print(f"Correctness FAILED: {e}")
        return False


# ============================================================
# 6. 单次 CUDA Graph 测量
# ============================================================

def benchmark_once(fn):

    torch.cuda.synchronize()

    ms = triton.testing.do_bench_cudagraph(
        fn,
        rep=CUDA_GRAPH_REP,
    )

    torch.cuda.synchronize()

    return ms * 1000


# ============================================================
# 7. 统计函数
# ============================================================

def calculate_statistics(samples):

    values = sorted(samples)
    n = len(values)

    # 最近秩法计算 P95
    p95_index = math.ceil(0.95 * n) - 1

    return {
        "mean_us": statistics.mean(values),
        "median_us": statistics.median(values),
        "p95_us": values[p95_index],
        "std_us": statistics.stdev(values) if n > 1 else 0.0,
        "min_us": values[0],
        "max_us": values[-1],
    }


# ============================================================
# 8. 获取候选配置
# ============================================================

def get_configs(m):

    if m >= 128:
        return PREFILL_CONFIGS

    return DECODE_CONFIGS


# ============================================================
# 9. 单个 Shape 的 Benchmark
# ============================================================

def benchmark_shape(M, K, raw_rows, summary_rows):

    N = K // 2

    print("\n" + "=" * 85)
    print(f"Shape: M={M}, K={K}, N={N}")
    print("=" * 85)

    x = torch.randn(
        (M, K),
        device="cuda",
        dtype=DTYPE,
    )

    # --------------------------------------------------------
    # PyTorch 参考结果
    # --------------------------------------------------------

    y_eager = swiglu_torch(x)
    y_compile = swiglu_compile(x)

    torch.cuda.synchronize()

    if not check_correctness(y_eager, y_compile):
        raise RuntimeError("torch.compile correctness failed")

    # --------------------------------------------------------
    # 构造待测任务
    # --------------------------------------------------------

    tasks = {}

    tasks["Eager"] = lambda: swiglu_torch(x)
    tasks["Compile"] = lambda: swiglu_compile(x)

    configs = get_configs(M)

    for block_size, num_warps in configs:

        name = f"BS{block_size}_W{num_warps}"

        tasks[name] = lambda bs=block_size, nw=num_warps: (
            swiglu_triton_config(x, bs, nw)
        )

    # --------------------------------------------------------
    # Triton 正确性检查
    # --------------------------------------------------------

    print("\nCorrectness Check:")
    print("-" * 85)

    valid_tasks = {
        "Eager",
        "Compile",
    }

    for block_size, num_warps in configs:

        name = f"BS{block_size}_W{num_warps}"

        try:
            y_test = tasks[name]()

            torch.cuda.synchronize()

            correct = check_correctness(
                y_eager,
                y_test,
            )

            if correct:
                valid_tasks.add(name)
                print(
                    f"{name:<18} PASS"
                )
            else:
                print(
                    f"{name:<18} FAILED"
                )

            del y_test

        except Exception as e:

            print(
                f"{name:<18} ERROR: {e}"
            )

        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # 预热所有有效任务
    #
    # 主要目的：
    # 1. 触发 Triton JIT 编译
    # 2. 触发 torch.compile 编译
    # 3. 避免首次编译时间进入测量
    # --------------------------------------------------------

    print("\nWarming up all valid configurations...")

    for name in valid_tasks:

        fn = tasks[name]

        for _ in range(WARMUP):
            fn()

        torch.cuda.synchronize()

    # --------------------------------------------------------
    # 多轮随机顺序测试
    # --------------------------------------------------------

    print("\nStarting randomized benchmark...")
    print(f"Rounds: {ROUNDS}")
    print(f"CUDA Graph rep: {CUDA_GRAPH_REP} ms")

    rng = random.Random(RANDOM_SEED + M)

    samples = {
        name: []
        for name in valid_tasks
    }

    task_names = list(valid_tasks)

    for round_id in range(ROUNDS):

        # 每轮随机打乱任务顺序
        rng.shuffle(task_names)

        for name in task_names:

            try:

                latency_us = benchmark_once(
                    tasks[name]
                )

                samples[name].append(latency_us)

            except Exception as e:

                print(
                    f"\nRound {round_id + 1}, "
                    f"{name} failed: {e}"
                )

                raise

        if (round_id + 1) % 5 == 0:

            print(
                f"Progress: {round_id + 1}/{ROUNDS}"
            )

    # --------------------------------------------------------
    # 统计结果
    # --------------------------------------------------------

    print("\nBenchmark Results:")
    print("-" * 85)

    compile_stats = calculate_statistics(
        samples["Compile"]
    )

    compile_median = compile_stats["median_us"]

    for name in sorted(samples):

        stats = calculate_statistics(
            samples[name]
        )

        speedup = compile_median / stats["median_us"]

        print(
            f"{name:<18}"
            f"Mean: {stats['mean_us']:>8.3f} us | "
            f"Median: {stats['median_us']:>8.3f} us | "
            f"P95: {stats['p95_us']:>8.3f} us | "
            f"Std: {stats['std_us']:>7.3f} us | "
            f"Speedup: {speedup:.4f}x"
        )

        # 汇总 CSV
        if name == "Eager":
            block_size = ""
            num_warps = ""
            category = "PyTorch Eager"

        elif name == "Compile":
            block_size = ""
            num_warps = ""
            category = "PyTorch Compile"

        else:
            category = "Triton"

            parts = name.replace("BS", "").split("_W")

            block_size = int(parts[0])
            num_warps = int(parts[1])

        summary_rows.append({
            "M": M,
            "K": K,
            "N": N,
            "category": category,
            "block_size": block_size,
            "num_warps": num_warps,
            "correctness": "PASS",
            "mean_us": stats["mean_us"],
            "median_us": stats["median_us"],
            "p95_us": stats["p95_us"],
            "std_us": stats["std_us"],
            "min_us": stats["min_us"],
            "max_us": stats["max_us"],
            "compile_median_us": compile_median,
            "speedup_vs_compile": speedup,
            "rounds": len(samples[name]),
        })

        # 原始 CSV：每轮单独保存
        for round_id, latency_us in enumerate(
            samples[name],
            start=1,
        ):

            raw_rows.append({
                "M": M,
                "K": K,
                "category": category,
                "config": name,
                "round": round_id,
                "latency_us": latency_us,
            })

    # --------------------------------------------------------
    # 输出 Triton 最佳候选配置
    # --------------------------------------------------------

    triton_names = [
        name for name in samples
        if name.startswith("BS")
    ]

    if triton_names:

        best_name = min(
            triton_names,
            key=lambda name: statistics.median(
                samples[name]
            ),
        )

        best_stats = calculate_statistics(
            samples[best_name]
        )

        best_speedup = (
            compile_median / best_stats["median_us"]
        )

        print("\nBest Triton Candidate:")
        print("-" * 85)

        print(f"Configuration : {best_name}")
        print(
            f"Median        : "
            f"{best_stats['median_us']:.3f} us"
        )
        print(
            f"P95           : "
            f"{best_stats['p95_us']:.3f} us"
        )
        print(
            f"Std           : "
            f"{best_stats['std_us']:.3f} us"
        )
        print(
            f"Speedup       : "
            f"{best_speedup:.4f}x"
        )

    del x, y_eager, y_compile
    torch.cuda.empty_cache()


# ============================================================
# 10. 保存 CSV
# ============================================================

def save_csv(path, rows, fieldnames):

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 11. 主函数
# ============================================================

def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 85)
    print("SwiGLU Stability Benchmark")
    print("=" * 85)

    print(f"GPU       : {torch.cuda.get_device_name()}")
    print(f"PyTorch   : {torch.__version__}")
    print(f"CUDA      : {torch.version.cuda}")
    print(f"Dtype     : {DTYPE}")
    print(f"Rounds    : {ROUNDS}")
    print(f"Warmup    : {WARMUP}")
    print(f"Graph rep : {CUDA_GRAPH_REP} ms")

    print("\nPrefill candidates:")
    print(PREFILL_CONFIGS)

    print("\nDecode candidates:")
    print(DECODE_CONFIGS)

    raw_rows = []
    summary_rows = []

    for M, K in SHAPES:

        benchmark_shape(
            M,
            K,
            raw_rows,
            summary_rows,
        )

    # --------------------------------------------------------
    # 保存原始数据
    # --------------------------------------------------------

    save_csv(
        RAW_CSV_PATH,
        raw_rows,
        [
            "M",
            "K",
            "category",
            "config",
            "round",
            "latency_us",
        ],
    )

    # --------------------------------------------------------
    # 保存汇总数据
    # --------------------------------------------------------

    save_csv(
        SUMMARY_CSV_PATH,
        summary_rows,
        [
            "M",
            "K",
            "N",
            "category",
            "block_size",
            "num_warps",
            "correctness",
            "mean_us",
            "median_us",
            "p95_us",
            "std_us",
            "min_us",
            "max_us",
            "compile_median_us",
            "speedup_vs_compile",
            "rounds",
        ],
    )

    # --------------------------------------------------------
    # 最终总结
    # --------------------------------------------------------

    print("\n" + "=" * 85)
    print("FINAL SUMMARY")
    print("=" * 85)

    for M, K in SHAPES:

        rows = [
            row for row in summary_rows
            if row["M"] == M
            and row["category"] == "Triton"
        ]

        if not rows:
            continue

        best = min(
            rows,
            key=lambda row: row["median_us"],
        )

        scene = "Prefill" if M >= 128 else "Decode"

        print(f"\n{scene} (M={M})")

        print(
            f"Best BLOCK_SIZE : {best['block_size']}"
        )
        print(
            f"Best num_warps  : {best['num_warps']}"
        )
        print(
            f"Median latency  : "
            f"{best['median_us']:.3f} us"
        )
        print(
            f"Compile latency : "
            f"{best['compile_median_us']:.3f} us"
        )
        print(
            f"Speedup         : "
            f"{best['speedup_vs_compile']:.4f}x"
        )

    print("\n" + "=" * 85)
    print("Benchmark finished.")
    print(f"Raw results     : {RAW_CSV_PATH}")
    print(f"Summary results : {SUMMARY_CSV_PATH}")
    print("=" * 85)


if __name__ == "__main__":
    main()
