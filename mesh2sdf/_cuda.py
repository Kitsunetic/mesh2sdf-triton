"""CUDA adapter for the Triton SDF kernels."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from ._distance import initialize_distance_grid
from ._sweep import sweep_distances
from ._triton import apply_signs


class CudaBackendUnavailableError(RuntimeError):
    """Raised when CUDA execution was requested without a CUDA device."""


def is_available() -> bool:
    return torch.cuda.is_available()


def _projected_sign_bounds(
    triangles: NDArray[np.float64],
    spacing: float,
    size: int,
) -> NDArray[np.int32]:
    scaled_yz = (triangles[:, :, 1:] + 1.0) / spacing
    lower = np.clip(np.ceil(scaled_yz.min(axis=1)), 0, size - 1).astype(np.int32)
    upper = np.clip(np.floor(scaled_yz.max(axis=1)), 0, size - 1).astype(np.int32)
    return np.stack((lower[:, 0], upper[:, 0], lower[:, 1], upper[:, 1]), axis=1)


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
    triangles = vertex_tensor[face_tensor].contiguous()
    spacing = float(np.float32(2.0 / size))
    triangle_array = vertices[faces.astype(np.int64, copy=False)].astype(np.float64)
    scaled = (triangle_array + 1.0) / spacing
    lower = np.clip(np.trunc(scaled.min(axis=1)).astype(np.int32) - 1, 0, size - 1)
    upper = np.clip(np.trunc(scaled.max(axis=1)).astype(np.int32) + 2, 0, size - 1)
    bounds_array = np.stack(
        (lower[:, 0], upper[:, 0], lower[:, 1], upper[:, 1], lower[:, 2], upper[:, 2]),
        axis=1,
    )
    bounds = torch.as_tensor(bounds_array, dtype=torch.int32, device=device)
    sign_bounds = torch.as_tensor(
        _projected_sign_bounds(triangle_array, spacing, size),
        dtype=torch.int32,
        device=device,
    )
    result, closest = initialize_distance_grid(triangles, bounds, size)
    sweep_distances(triangles, result, closest, size)
    apply_signs(triangles, sign_bounds, result, size)
    return result.reshape(size, size, size).cpu().numpy()
