"""Autograd entry point for the Triton SDF implementation."""

from __future__ import annotations

import torch
from torch import Tensor

from ._backward import accumulate_vertex_gradients
from ._cuda import compute_cuda_tensor


class _TritonSDF(torch.autograd.Function):
    """Run the existing Triton forward kernels with a custom vertex backward."""

    @staticmethod
    def forward(ctx, vertices: Tensor, faces: Tensor, size: int) -> Tensor:
        field, closest = compute_cuda_tensor(vertices, faces, size)
        ctx.save_for_backward(vertices, faces, closest, field)
        return field

    @staticmethod
    def backward(ctx, *output_gradients: Tensor) -> tuple[Tensor, None, None]:
        (output_gradient,) = output_gradients
        vertices, faces, closest, field = ctx.saved_tensors
        vertex_gradients = accumulate_vertex_gradients(
            vertices, faces, closest, field, output_gradient.contiguous()
        )
        return vertex_gradients, None, None


def compute_triton(vertices: Tensor, faces: Tensor, size: int = 128) -> Tensor:
    """Compute a CUDA SDF with a piecewise vertex subgradient.

    The backward treats the forward-selected closest feature and parity sign as
    constants. Samples with zero unsigned distance receive a zero subgradient.
    """
    normalized_faces = faces.to(device=vertices.device, dtype=torch.long)
    return _TritonSDF.apply(vertices, normalized_faces, size)
