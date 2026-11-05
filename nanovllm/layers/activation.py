import sys
import os
sys.path.insert(0, "/home/chen17/nano-vllm")

import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.layers.triton_kernels import silu_and_mul_triton


_TRACE_SEEN_SHAPES = set()


class SiluAndMul(nn.Module):

    @torch.compile
    def forward_compile(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up

    def forward_eager(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if os.getenv("NANOVLLM_TRACE_SILU") == "1":
            shape = tuple(x.shape)
            if shape not in _TRACE_SEEN_SHAPES:
                _TRACE_SEEN_SHAPES.add(shape)
                print(
                    f"[SwiGLU] shape={shape}, "
                    f"dtype={x.dtype}, "
                    f"device={x.device}"
                )

        # 优先使用新的三路切换变量
        impl = os.getenv("NANOVLLM_SILU_IMPL")

        # 兼容旧环境变量，避免影响之前的使用方式
        if impl is None:
            use_triton = (
                os.getenv("NANOVLLM_USE_TRITON_SILU", "1") == "1"
            )
            impl = "triton" if use_triton else "compile"

        if impl == "eager":
            return self.forward_eager(x)

        elif impl == "compile":
            return self.forward_compile(x)

        elif impl == "triton":
            return silu_and_mul_triton(x)

        else:
            raise ValueError(
                f"Unsupported NANOVLLM_SILU_IMPL={impl}. "
                "Expected: eager, compile, triton."
            )
