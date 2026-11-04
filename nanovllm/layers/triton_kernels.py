
import torch
import triton
import triton.language as tl


@triton.jit
def _silu_and_mul_kernel(
    X,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M * N

    row = offsets // N
    col = offsets % N

    # X 的每一行布局：[gate | up]
    gate = tl.load(
        X + row * (2 * N) + col,
        mask=mask,
        other=0,
    ).to(tl.float32)

    up = tl.load(
        X + row * (2 * N) + col + N,
        mask=mask,
        other=0,
    ).to(tl.float32)

    # SiLU(gate) * up
    silu = gate / (1.0 + tl.exp(-gate))
    output = silu * up

    tl.store(Y + offsets, output, mask=mask)


def silu_and_mul_triton(x: torch.Tensor) -> torch.Tensor:
    """
    x: [..., 2 * N]
    output: [..., N]
    """
    if not x.is_cuda:
        raise ValueError("Triton SwiGLU requires a CUDA tensor")

    if not x.is_contiguous():
        raise ValueError("SwiGLU input must be contiguous")

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

    block_size = 1024
    grid = (triton.cdiv(m * n, block_size),)

    _silu_and_mul_kernel[grid](
        x_2d,
        output,
        m,
        n,
        block_size,
    )

    return output
