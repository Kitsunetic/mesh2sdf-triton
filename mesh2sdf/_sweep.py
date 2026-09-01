"""Wavefront implementation of the original eight-direction fast sweep."""

from __future__ import annotations

from collections.abc import Callable

import triton
import triton.language as tl
from torch import Tensor

from ._triton import _segment_distance_squared


@triton.jit
def _triangle_distance(triangles, triangle_ids, valid, px, py, pz) -> tl.tensor:
    base = triangle_ids * 9
    ax = tl.load(triangles + base, mask=valid, other=0.0)
    ay = tl.load(triangles + base + 1, mask=valid, other=0.0)
    az = tl.load(triangles + base + 2, mask=valid, other=0.0)
    bx = tl.load(triangles + base + 3, mask=valid, other=0.0)
    by = tl.load(triangles + base + 4, mask=valid, other=0.0)
    bz = tl.load(triangles + base + 5, mask=valid, other=0.0)
    cx = tl.load(triangles + base + 6, mask=valid, other=0.0)
    cy = tl.load(triangles + base + 7, mask=valid, other=0.0)
    cz = tl.load(triangles + base + 8, mask=valid, other=0.0)
    x13x, x13y, x13z = ax - cx, ay - cy, az - cz
    x23x, x23y, x23z = bx - cx, by - cy, bz - cz
    x03x, x03y, x03z = px - cx, py - cy, pz - cz
    m13 = x13x * x13x + x13y * x13y + x13z * x13z
    m23 = x23x * x23x + x23y * x23y + x23z * x23z
    product = x13x * x23x + x13y * x23y + x13z * x23z
    inverse = 1.0 / tl.maximum(m13 * m23 - product * product, 1e-30)
    a = x13x * x03x + x13y * x03y + x13z * x03z
    b = x23x * x03x + x23y * x03y + x23z * x03z
    w23 = inverse * (m23 * a - product * b)
    w31 = inverse * (m13 * b - product * a)
    w12 = 1.0 - w23 - w31
    qx = w23 * ax + w31 * bx + w12 * cx
    qy = w23 * ay + w31 * by + w12 * cy
    qz = w23 * az + w31 * bz + w12 * cz
    dx, dy, dz = px - qx, py - qy, pz - qz
    plane = dx * dx + dy * dy + dz * dz
    ab = _segment_distance_squared(px, py, pz, ax, ay, az, bx, by, bz)
    ac = _segment_distance_squared(px, py, pz, ax, ay, az, cx, cy, cz)
    bc = _segment_distance_squared(px, py, pz, bx, by, bz, cx, cy, cz)
    edge = tl.where(w23 > 0.0, tl.minimum(ab, ac),
                    tl.where(w31 > 0.0, tl.minimum(ab, bc), tl.minimum(ac, bc)))
    inside = (w23 >= 0.0) & (w31 >= 0.0) & (w12 >= 0.0)
    squared = tl.where(inside, plane, edge)
    return tl.where(valid, tl.sqrt(tl.maximum(squared, 0.0)), float("inf"))


@triton.jit
def _sweep_kernel(
    triangles, distances, closest, size: tl.constexpr, diagonal,
    di: tl.constexpr, dj: tl.constexpr, dk: tl.constexpr, block: tl.constexpr,
):
    ids = tl.program_id(0) * block + tl.arange(0, block)
    edge = size - 1
    transformed_i = ids // edge + 1
    transformed_j = ids % edge + 1
    transformed_k = diagonal - transformed_i - transformed_j
    valid = (ids < edge * edge) & (transformed_k >= 1) & (transformed_k < size)
    i = tl.where(di > 0, transformed_i, size - 1 - transformed_i)
    j = tl.where(dj > 0, transformed_j, size - 1 - transformed_j)
    k = tl.where(dk > 0, transformed_k, size - 1 - transformed_k)
    spacing = 2.0 / size
    px = -1.0 + i.to(tl.float32) * spacing
    py = -1.0 + j.to(tl.float32) * spacing
    pz = -1.0 + k.to(tl.float32) * spacing
    offset = (i * size + j) * size + k
    best = tl.load(distances + offset, mask=valid, other=36.0)
    best_triangle = tl.load(closest + offset, mask=valid, other=-1)
    for neighbor in range(1, 8):
        ni = i - di * (neighbor & 1)
        nj = j - dj * ((neighbor >> 1) & 1)
        nk = k - dk * ((neighbor >> 2) & 1)
        neighbor_offset = (ni * size + nj) * size + nk
        candidate = tl.load(closest + neighbor_offset, mask=valid, other=-1)
        candidate_distance = _triangle_distance(
            triangles, candidate, valid & (candidate >= 0), px, py, pz
        )
        use_candidate = candidate_distance < best
        best = tl.where(use_candidate, candidate_distance, best)
        best_triangle = tl.where(use_candidate, candidate, best_triangle)
    tl.store(distances + offset, best, mask=valid)
    tl.store(closest + offset, best_triangle, mask=valid)


def _launch(
    kernel: Callable[..., None], triangles: Tensor, distances: Tensor,
    closest: Tensor, size: int, diagonal: int, direction: tuple[int, int, int],
) -> None:
    kernel(triangles, distances, closest, size, diagonal, *direction, 128,
           enable_fp_fusion=False)


def sweep_distances(
    triangles: Tensor, distances: Tensor, closest: Tensor, size: int,
) -> None:
    directions = (
        (1, 1, 1), (-1, -1, -1), (1, 1, -1), (-1, -1, 1),
        (1, -1, 1), (-1, 1, -1), (1, -1, -1), (-1, 1, 1),
    )
    grid = ((size - 1) * (size - 1) + 127) // 128
    for _ in range(2):
        for direction in directions:
            for diagonal in range(3, 3 * size - 2):
                kernel: Callable[..., None] = _sweep_kernel[(grid,)]
                _launch(kernel, triangles, distances, closest,
                        size, diagonal, direction)
