import sys
sys.path.insert(0, "/home/chen17/nano-vllm")
import torch

from nanovllm.layers.activation import SiluAndMul


def test_silu_and_mul():
    assert torch.cuda.is_available()

    layer = SiluAndMul().cuda()

    test_shapes = [
        (1, 4096),
        (4, 4096),
        (16, 4096),
        (32, 4096),
        (4, 8192),
        (16, 8192),
    ]

    dtypes = [
        torch.float16,
        torch.float32,
    ]

    for shape in test_shapes:
        for dtype in dtypes:
            x = torch.randn(
                *shape,
                device="cuda",
                dtype=dtype,
            )

            expected = layer.forward_torch(x)
            actual = layer(x)

            if dtype == torch.float16:
                atol, rtol = 1e-3, 1e-3
            else:
                atol, rtol = 1e-5, 1e-5

            torch.testing.assert_close(
                actual,
                expected,
                atol=atol,
                rtol=rtol,
            )

            print(
                f"PASS shape={shape}, "
                f"dtype={dtype}, "
                f"max_error={(actual - expected).abs().max().item():.6e}"
            )


if __name__ == "__main__":
    test_silu_and_mul()
