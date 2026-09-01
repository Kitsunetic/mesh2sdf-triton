"""Wavefront implementation of the original eight-direction fast sweep."""

from __future__ import annotations

from collections.abc import Callable

import torch
import triton
import triton.language as tl
from torch import Tensor

from ._triton import _segment_distance_squared

SWEEP_TOLERANCE = 0.0


@triton.jit
def _triangle_distance(triangles, triangle_ids, valid, px, py, pz):
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
def _update(triangles, candidate, valid, px, py, pz,
            best, best_triangle, tolerance):
    candidate_distance = _triangle_distance(
        triangles, candidate, valid & (candidate >= 0), px, py, pz
    )
    use_candidate = candidate_distance < best - tolerance
    return (
        tl.where(use_candidate, candidate_distance, best),
        tl.where(use_candidate, candidate, best_triangle),
    )


@triton.jit
def _sweep_kernel(
    triangles,
    distances,
    closest,
    size: tl.constexpr,
    diagonal,
    di: tl.constexpr,
    dj: tl.constexpr,
    dk: tl.constexpr,
    block: tl.constexpr,
    tolerance,
):
    lanes = tl.arange(0, block)
    transformed_j = tl.maximum(1, diagonal - (size - 1)) + lanes
    transformed_k = diagonal - transformed_j
    valid = (transformed_j < size) & (transformed_k >= 1) & (transformed_k < size)
    j = tl.where(dj > 0, transformed_j, size - 1 - transformed_j)
    k = tl.where(dk > 0, transformed_k, size - 1 - transformed_k)
    spacing = 2.0 / size
    py = -1.0 + j.to(tl.float32) * spacing
    pz = -1.0 + k.to(tl.float32) * spacing
    boundary_i = tl.where(di > 0, 0, size - 1)
    previous_offset = (boundary_i * size + j) * size + k
    previous_triangle = tl.load(closest + previous_offset, mask=valid, other=-1)

    for transformed_i in range(1, size):
        i = tl.where(di > 0, transformed_i, size - 1 - transformed_i)
        px = -1.0 + i.to(tl.float32) * spacing
        offset = (i * size + j) * size + k
        best = tl.load(distances + offset, mask=valid, other=36.0)
        best_triangle = tl.load(closest + offset, mask=valid, other=-1)
        j_previous, k_previous = j - dj, k - dk
        best, best_triangle = _update(
            triangles, previous_triangle, valid, px, py, pz,
            best, best_triangle, tolerance,
        )
        candidate = tl.load(
            closest + (i * size + j_previous) * size + k, mask=valid, other=-1
        )
        best, best_triangle = _update(
            triangles, candidate, valid, px, py, pz, best, best_triangle, tolerance
        )
        candidate = tl.load(
            closest + ((i - di) * size + j_previous) * size + k,
            mask=valid, other=-1,
        )
        best, best_triangle = _update(
            triangles, candidate, valid, px, py, pz, best, best_triangle, tolerance
        )
        candidate = tl.load(
            closest + (i * size + j) * size + k_previous, mask=valid, other=-1
        )
        best, best_triangle = _update(
            triangles, candidate, valid, px, py, pz, best, best_triangle, tolerance
        )
        candidate = tl.load(
            closest + ((i - di) * size + j) * size + k_previous,
            mask=valid, other=-1,
        )
        best, best_triangle = _update(
            triangles, candidate, valid, px, py, pz, best, best_triangle, tolerance
        )
        candidate = tl.load(
            closest + (i * size + j_previous) * size + k_previous,
            mask=valid, other=-1,
        )
        best, best_triangle = _update(
            triangles, candidate, valid, px, py, pz, best, best_triangle, tolerance
        )
        candidate = tl.load(
            closest + ((i - di) * size + j_previous) * size + k_previous,
            mask=valid, other=-1,
        )
        best, best_triangle = _update(
            triangles, candidate, valid, px, py, pz, best, best_triangle, tolerance
        )
        tl.store(distances + offset, best, mask=valid)
        tl.store(closest + offset, best_triangle, mask=valid)
        previous_triangle = best_triangle


def _launch(
    kernel: Callable[..., None], triangles: Tensor, distances: Tensor,
    closest: Tensor, size: int, diagonal: int, direction: tuple[int, int, int],
) -> None:
    kernel(triangles, distances, closest, size, diagonal, *direction, 128,
           SWEEP_TOLERANCE, enable_fp_fusion=False)


def sweep_distances(
    triangles: Tensor, distances: Tensor, closest: Tensor, size: int,
) -> None:
    directions = (
        (1, 1, 1), (-1, -1, -1), (1, 1, -1), (-1, -1, 1),
        (1, -1, 1), (-1, 1, -1), (1, -1, -1), (-1, 1, 1),
    )
    block = triton.next_power_of_2(size)
    for _ in range(2):
        for direction in directions:
            for diagonal in range(2, 2 * size - 1):
                kernel: Callable[..., None] = _sweep_kernel[(1,)]
                _launch(kernel, triangles, distances, closest,
                        size, diagonal, direction)
