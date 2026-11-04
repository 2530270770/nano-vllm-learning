import sys
sys.path.insert(0, "/home/chen17/nano-vllm")


import torch
import triton
import torch.nn.functional as F

from nanovllm.layers.triton_kernels import silu_and_mul_triton


def pytorch_silu_and_mul(x):
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


def benchmark(fn, rep=100):
    # 预热
    for _ in range(20):
        fn()

    torch.cuda.synchronize()

    # CUDA Graph benchmark，减少 Python 调度开销
    if hasattr(triton.testing, "do_bench_cudagraph"):
        return triton.testing.do_bench_cudagraph(
            fn,
            rep=rep,
        )

    return triton.testing.do_bench(fn, rep=rep)


def main():
    assert torch.cuda.is_available()

    print("GPU:", torch.cuda.get_device_name())

    shapes = [
        (1, 4096),
        (4, 4096),
        (16, 4096),
        (32, 4096),
        (4, 8192),
        (16, 8192),
        (2048, 65536),
    ]

    for dtype in [torch.float16, torch.float32]:
        print(f"\n========== dtype={dtype} ==========")

        for m, n in shapes:
            x = torch.randn(
                (m, 2 * n),
                device="cuda",
                dtype=dtype,
            )

            torch_fn = lambda: pytorch_silu_and_mul(x)
            triton_fn = lambda: silu_and_mul_triton(x)

            torch_ms = benchmark(torch_fn)
            triton_ms = benchmark(triton_fn)

            speedup = torch_ms / triton_ms

            print(
                f"M={m:5d}, N={n:6d} | "
                f"PyTorch={torch_ms * 1000:9.3f} us | "
                f"Triton={triton_ms * 1000:9.3f} us | "
                f"Speedup={speedup:.2f}x"
            )


if __name__ == "__main__":
    main()
