import sys
import os
sys.path.insert(0, "/home/chen17/nano-vllm")

_TRACE_SEEN_SHAPES = set()
import torch
from torch import nn
import torch.nn.functional as F
from nanovllm.layers.triton_kernels import silu_and_mul_triton

class SiluAndMul(nn.Module):

    @torch.compile
    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

    def forward(self, x):
        if os.getenv("NANOVLLM_TRACE_SILU") == "1":
            shape = tuple(x.shape)
            if shape not in _TRACE_SEEN_SHAPES:
                _TRACE_SEEN_SHAPES.add(shape)
                print(
                    f"[SwiGLU] shape={shape}, "
                    f"dtype={x.dtype}, "
                    f"device={x.device}"
                )

        use_triton = os.getenv("NANOVLLM_USE_TRITON_SILU", "1") == "1"

        if use_triton:
            return silu_and_mul_triton(x)

        return self.forward_torch(x)