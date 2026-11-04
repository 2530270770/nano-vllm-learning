import sys
sys.path.insert(0, "/home/chen17/nano-vllm")
import torch
import torch.nn.functional as F
import triton.testing

from nanovllm.layers.triton_kernels import silu_and_mul_triton


# 与 Qwen3-0.6B 实际运行时的 Shape 对齐
SHAPES = [
    (512, 6144),  # Prefill: 4 * 128 tokens
    (4, 6144),    # Decode: 4 requests
]

DTYPE = torch.bfloat16
WARMUP = 10
REPEATS = 5


def swiglu_torch(x):
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


# PyTorch Compile 版本
swiglu_compile = torch.compile(swiglu_torch)


def benchmark(fn):
    # 先执行，完成编译和必要的初始化
    for _ in range(WARMUP):
        fn()

    torch.cuda.synchronize()

    # 使用 CUDA Graph 降低 Python 调用开销的干扰
    results = []
    for _ in range(REPEATS):
        ms = triton.testing.do_bench_cudagraph(fn, rep=100)
        results.append(ms)

    return sorted(results)[len(results) // 2]


def main():
    print("=" * 65)
    print("SwiGLU Three-Way Benchmark")
    print("=" * 65)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Dtype: {DTYPE}")
    print()

    for M, K in SHAPES:
        N = K // 2

        print("-" * 65)
        print(f"Shape: M={M}, K={K}, N={N}")

        x = torch.randn(
            (M, K),
            device="cuda",
            dtype=DTYPE,
        )

        # 先验证三种实现的正确性
        y_eager = swiglu_torch(x)
        y_compile = swiglu_compile(x)
        y_triton = silu_and_mul_triton(x)

        torch.cuda.synchronize()

        torch.testing.assert_close(
            y_compile, y_eager, rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(
            y_triton, y_eager, rtol=2e-2, atol=2e-2
        )

        print("Correctness: PASS")

        # 三种实现分别测试
        eager_ms = benchmark(lambda: swiglu_torch(x))
        compile_ms = benchmark(lambda: swiglu_compile(x))
        triton_ms = benchmark(lambda: silu_and_mul_triton(x))

        print(f"PyTorch Eager  : {eager_ms * 1000:.2f} us")
        print(f"PyTorch Compile: {compile_ms * 1000:.2f} us")
        print(f"Triton Fused   : {triton_ms * 1000:.2f} us")

        print()
        print("相对 PyTorch Compile：")
        print(f"Triton speedup: {compile_ms / triton_ms:.3f}x")

        del x, y_eager, y_compile, y_triton
        torch.cuda.empty_cache()

    print("=" * 65)
    print("Benchmark finished.")


if __name__ == "__main__":
    main()