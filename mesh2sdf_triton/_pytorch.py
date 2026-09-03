"""Native PyTorch oracle for the original Mesh2SDF construction."""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor

from ._pytorch_sign import apply_signs

_EXACT_BAND: Final = 1
_DIRECTIONS: Final = (
    (1, 1, 1),
    (-1, -1, -1),
    (1, 1, -1),
    (-1, -1, 1),
    (1, -1, 1),
    (-1, 1, -1),
    (1, -1, -1),
    (-1, 1, 1),
)


def _dot(left: Tensor, right: Tensor) -> Tensor:
    return (
        left[..., 0] * right[..., 0]
        + left[..., 1] * right[..., 1]
        + left[..., 2] * right[..., 2]
    )


def _segment_distance(points: Tensor, start: Tensor, end: Tensor) -> Tensor:
    direction = end - start
    length_squared = _dot(direction, direction)
    parameter = _dot(end - points, direction) / length_squared
    parameter = parameter.clamp(0.0, 1.0)
    closest = parameter.unsqueeze(-1) * start + (1.0 - parameter).unsqueeze(-1) * end
    offset = points - closest
    return torch.sqrt(_dot(offset, offset))


def _triangle_distance(points: Tensor, triangles: Tensor) -> Tensor:
    first, second, third = triangles.unbind(dim=-2)
    first_third = first - third
    second_third = second - third
    point_third = points - third
    first_squared = _dot(first_third, first_third)
    second_squared = _dot(second_third, second_third)
    first_second = _dot(first_third, second_third)
    inverse = 1.0 / torch.maximum(
        first_squared * second_squared - first_second * first_second,
        torch.full_like(first_squared, 1e-30),
    )
    first_point = _dot(first_third, point_third)
    second_point = _dot(second_third, point_third)
    second_third_weight = inverse * (
        second_squared * first_point - first_second * second_point
    )
    third_first_weight = inverse * (
        first_squared * second_point - first_second * first_point
    )
    first_second_weight = 1.0 - second_third_weight - third_first_weight
    plane_point = (
        second_third_weight.unsqueeze(-1) * first
        + third_first_weight.unsqueeze(-1) * second
        + first_second_weight.unsqueeze(-1) * third
    )
    plane_offset = points - plane_point
    plane_distance = torch.sqrt(_dot(plane_offset, plane_offset))
    first_second_distance = _segment_distance(points, first, second)
    first_third_distance = _segment_distance(points, first, third)
    second_third_distance = _segment_distance(points, second, third)
    edge_distance = torch.where(
        second_third_weight > 0.0,
        torch.minimum(first_second_distance, first_third_distance),
        torch.where(
            third_first_weight > 0.0,
            torch.minimum(first_second_distance, second_third_distance),
            torch.minimum(first_third_distance, second_third_distance),
        ),
    )
    inside = (
        (second_third_weight >= 0.0)
        & (third_first_weight >= 0.0)
        & (first_second_weight >= 0.0)
    )
    return torch.where(inside, plane_distance, edge_distance)


def _triangle_bounds(
    triangle: Tensor, size: int, spacing: Tensor
) -> tuple[int, int, int, int, int, int]:
    coordinates = (triangle.to(torch.float64) + 1.0) / spacing.to(torch.float64)
    minimum, maximum = torch.aminmax(coordinates, dim=0)
    lower = (torch.trunc(minimum).to(torch.int64) - _EXACT_BAND).clamp(0, size - 1)
    upper = (torch.trunc(maximum).to(torch.int64) + _EXACT_BAND + 1).clamp(0, size - 1)
    return (
        int(lower[0]),
        int(upper[0]),
        int(lower[1]),
        int(upper[1]),
        int(lower[2]),
        int(upper[2]),
    )


def _initialize_distances(
    triangles: Tensor, size: int, spacing: Tensor
) -> tuple[Tensor, Tensor]:
    point_count = size**3
    distances = torch.full(
        (point_count,),
        float(3 * size) * float(spacing),
        dtype=triangles.dtype,
        device=triangles.device,
    )
    closest = torch.full((point_count,), -1, dtype=torch.int64, device=triangles.device)
    for triangle_id, triangle in enumerate(triangles):
        i0, i1, j0, j1, k0, k1 = _triangle_bounds(triangle, size, spacing)
        i, j, k = torch.meshgrid(
            torch.arange(i0, i1 + 1, device=triangles.device),
            torch.arange(j0, j1 + 1, device=triangles.device),
            torch.arange(k0, k1 + 1, device=triangles.device),
            indexing="ij",
        )
        points = torch.stack((i, j, k), dim=-1).to(triangles.dtype) * spacing - 1.0
        offsets = (i * size * size + j * size + k).reshape(-1)
        candidate = _triangle_distance(points.reshape(-1, 3), triangle).reshape(-1)
        previous = distances[offsets]
        is_closer = candidate < previous
        distances = distances.index_copy(
            0, offsets, torch.where(is_closer, candidate, previous)
        )
        _ = closest.index_copy_(
            0,
            offsets,
            torch.where(
                is_closer, torch.full_like(offsets, triangle_id), closest[offsets]
            ),
        )
    return distances, closest


def _sweep(
    distances: Tensor, closest: Tensor, triangles: Tensor, size: int, spacing: Tensor
) -> Tensor:
    transformed = torch.arange(1, size, device=triangles.device)
    transformed_i, transformed_j = torch.meshgrid(
        transformed, transformed, indexing="ij"
    )
    for _ in range(2):
        for di, dj, dk in _DIRECTIONS:
            for diagonal in range(3, 3 * size - 2):
                transformed_k = diagonal - transformed_i - transformed_j
                valid = (transformed_k >= 1) & (transformed_k < size)
                ti = transformed_i[valid]
                tj = transformed_j[valid]
                tk = transformed_k[valid]
                i = ti if di > 0 else size - 1 - ti
                j = tj if dj > 0 else size - 1 - tj
                k = tk if dk > 0 else size - 1 - tk
                offsets = i * size * size + j * size + k
                points = (
                    torch.stack((i, j, k), dim=-1).to(triangles.dtype) * spacing - 1.0
                )
                best = distances[offsets]
                best_triangle = closest[offsets]
                for neighbor in range(1, 8):
                    ni = i - di * (neighbor & 1)
                    nj = j - dj * ((neighbor >> 1) & 1)
                    nk = k - dk * ((neighbor >> 2) & 1)
                    neighbor_offsets = ni * size * size + nj * size + nk
                    candidate_triangle = closest[neighbor_offsets]
                    candidate_distance = _triangle_distance(
                        points, triangles[candidate_triangle.clamp_min(0)]
                    )
                    candidate_distance = torch.where(
                        candidate_triangle >= 0,
                        candidate_distance,
                        torch.full_like(candidate_distance, float("inf")),
                    )
                    is_closer = candidate_distance < best
                    best = torch.where(is_closer, candidate_distance, best)
                    best_triangle = torch.where(
                        is_closer, candidate_triangle, best_triangle
                    )
                distances = distances.index_copy(0, offsets, best)
                _ = closest.index_copy_(0, offsets, best_triangle)
    return distances


def compute_pytorch(vertices: Tensor, faces: Tensor, size: int = 128) -> Tensor:
    """Compute the original SDFGen-style field with PyTorch operations only.

    ``vertices`` must be a normalized float32 tensor; ``faces`` indexes its rows.
    The result remains connected to ``vertices`` in PyTorch Autograd, except at
    the ordinary closest-feature and parity discontinuities of an SDF.
    """
    triangles = vertices[faces.to(dtype=torch.long)]
    spacing = torch.tensor(2.0 / size, dtype=vertices.dtype, device=vertices.device)
    distances, closest = _initialize_distances(triangles, size, spacing)
    distances = _sweep(distances, closest, triangles, size, spacing)
    return apply_signs(distances, triangles, size, spacing).reshape(size, size, size)
