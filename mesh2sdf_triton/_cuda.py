"""CUDA adapter for the Triton SDF kernels."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ._distance import initialize_distance_grid
from ._sweep import sweep_distances
from ._triton import apply_signs


class CudaBackendUnavailableError(RuntimeError):
    """Raised when CUDA execution was requested without a CUDA device."""


def is_available() -> bool:
    return torch.cuda.is_available()


def _triangle_bounds(triangles: Tensor, size: int) -> tuple[Tensor, Tensor]:
    spacing = float(np.float32(2.0 / size))
    scaled = (triangles.to(torch.float64) + 1.0) / spacing
    minimum, maximum = torch.aminmax(scaled, dim=1)

    lower = (torch.trunc(minimum).to(torch.int32) - 1).clamp(0, size - 1)
    upper = (torch.trunc(maximum).to(torch.int32) + 2).clamp(0, size - 1)
    distance = torch.stack(
        (lower[:, 0], upper[:, 0], lower[:, 1], upper[:, 1], lower[:, 2], upper[:, 2]),
        dim=1,
    )

    sign_lower = torch.ceil(minimum[:, 1:]).clamp(0, size - 1).to(torch.int32)
    sign_upper = torch.floor(maximum[:, 1:]).clamp(0, size - 1).to(torch.int32)
    sign = torch.stack(
        (sign_lower[:, 0], sign_upper[:, 0], sign_lower[:, 1], sign_upper[:, 1]),
        dim=1,
    )
    return distance, sign


def compute_cuda(
    vertices: NDArray[np.float32],
    faces: NDArray[np.uint32],
    size: int,
    device: str,
) -> NDArray[np.float32]:
    if not is_available():
        raise CudaBackendUnavailableError("CUDA is not available")
    vertex_tensor = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    face_tensor = torch.as_tensor(faces.astype(np.int64, copy=False), device=device)
    result, _ = compute_cuda_tensor(vertex_tensor, face_tensor, size)
    return result.cpu().numpy()


def compute_cuda_tensor(
    vertices: Tensor, faces: Tensor, size: int
) -> tuple[Tensor, Tensor]:
    """Run the forward kernels and retain closest-triangle IDs for backward."""
    if not is_available() or not vertices.is_cuda:
        raise CudaBackendUnavailableError("a CUDA vertex tensor is required")
    face_tensor = faces.to(device=vertices.device, dtype=torch.long)
    triangles = vertices[face_tensor].contiguous()
    bounds, sign_bounds = _triangle_bounds(triangles, size)
    result, closest = initialize_distance_grid(triangles, bounds, size)
    sweep_distances(triangles, result, closest, size)
    _ = apply_signs(triangles, sign_bounds, result, size)
    return result.reshape(size, size, size), closest
