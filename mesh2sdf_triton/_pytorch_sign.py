"""PyTorch parity-sign pass for the SDF reference implementation."""

from __future__ import annotations

import torch
from torch import Tensor


def _orientation(
    first_x: Tensor, first_y: Tensor, second_x: Tensor, second_y: Tensor
) -> Tensor:
    area = first_y * second_x - first_x * second_y
    positive = torch.ones_like(area, dtype=torch.int64)
    negative = -positive
    return torch.where(
        area > 0.0,
        positive,
        torch.where(
            area < 0.0,
            negative,
            torch.where(
                second_y > first_y,
                positive,
                torch.where(
                    second_y < first_y,
                    negative,
                    torch.where(
                        first_x > second_x,
                        positive,
                        torch.where(
                            first_x < second_x, negative, torch.zeros_like(positive)
                        ),
                    ),
                ),
            ),
        ),
    )


def apply_signs(
    distances: Tensor, triangles: Tensor, size: int, spacing: Tensor
) -> Tensor:
    """Apply the original projected-triangle parity convention to distances."""
    counts = torch.zeros((size, size, size), dtype=torch.int64, device=triangles.device)
    grid_spacing = spacing.to(torch.float64)
    for triangle in triangles.to(torch.float64):
        scaled = (triangle + 1.0) / grid_spacing
        j0 = max(int(torch.ceil(scaled[:, 1].min())), 0)
        j1 = min(int(torch.floor(scaled[:, 1].max())), size - 1)
        k0 = max(int(torch.ceil(scaled[:, 2].min())), 0)
        k1 = min(int(torch.floor(scaled[:, 2].max())), size - 1)
        if j0 > j1 or k0 > k1:
            continue
        j, k = torch.meshgrid(
            torch.arange(j0, j1 + 1, device=triangles.device),
            torch.arange(k0, k1 + 1, device=triangles.device),
            indexing="ij",
        )
        first, second, third = scaled.unbind(dim=0)
        first_y, first_z = first[1] - j, first[2] - k
        second_y, second_z = second[1] - j, second[2] - k
        third_y, third_z = third[1] - j, third[2] - k
        first_weight = second_z * third_y - second_y * third_z
        second_weight = third_z * first_y - third_y * first_z
        third_weight = first_z * second_y - first_y * second_z
        first_sign = _orientation(second_y, second_z, third_y, third_z)
        second_sign = _orientation(third_y, third_z, first_y, first_z)
        third_sign = _orientation(first_y, first_z, second_y, second_z)
        inside = (
            (first_sign != 0) & (second_sign == first_sign) & (third_sign == first_sign)
        )
        total = first_weight + second_weight + third_weight
        denominator = torch.where(inside, total, torch.ones_like(total))
        intersection = (
            first_weight / denominator * first[0]
            + second_weight / denominator * second[0]
            + third_weight / denominator * third[0]
        )
        interval = torch.ceil(intersection).to(torch.int64)
        valid = inside & (interval < size)
        offsets = interval.clamp_min(0) * size * size + j * size + k
        flat_counts = counts.reshape(-1)
        valid_offsets = offsets[valid]
        _ = flat_counts.index_add_(0, valid_offsets, torch.ones_like(valid_offsets))
    parity = torch.remainder(torch.cumsum(counts, dim=0), 2) == 1
    return torch.where(parity.reshape(-1), -distances, distances)
