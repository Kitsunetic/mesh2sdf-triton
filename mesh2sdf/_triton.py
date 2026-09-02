"""Triton kernels for applying signed-distance parity."""

from __future__ import annotations

from collections.abc import Callable

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _orientation(x1, y1, x2, y2):
    area = y1 * x2 - x1 * y2
    return tl.where(
        area > 0.0,
        1,
        tl.where(
            area < 0.0,
            -1,
            tl.where(
                y2 > y1,
                1,
                tl.where(y2 < y1, -1, tl.where(x1 > x2, 1, tl.where(x1 < x2, -1, 0))),
            ),
        ),
    )


@triton.jit
def _intersection_kernel(
    triangles,
    counts,
    n_triangles: tl.constexpr,
    size: tl.constexpr,
    block: tl.constexpr,
):
    pair_ids = tl.program_id(0) * block + tl.arange(0, block)
    lines = size * size
    triangle_ids = pair_ids // lines
    line_ids = pair_ids - triangle_ids * lines
    valid = triangle_ids < n_triangles
    base = triangle_ids * 9
    ax = tl.load(triangles + base, mask=valid, other=0.0).to(tl.float64)
    ay = tl.load(triangles + base + 1, mask=valid, other=0.0).to(tl.float64)
    az = tl.load(triangles + base + 2, mask=valid, other=0.0).to(tl.float64)
    bx = tl.load(triangles + base + 3, mask=valid, other=0.0).to(tl.float64)
    by = tl.load(triangles + base + 4, mask=valid, other=0.0).to(tl.float64)
    bz = tl.load(triangles + base + 5, mask=valid, other=0.0).to(tl.float64)
    cx = tl.load(triangles + base + 6, mask=valid, other=0.0).to(tl.float64)
    cy = tl.load(triangles + base + 7, mask=valid, other=0.0).to(tl.float64)
    cz = tl.load(triangles + base + 8, mask=valid, other=0.0).to(tl.float64)
    spacing = tl.full((), 2.0 / size, tl.float32).to(tl.float64)
    py = (line_ids // size).to(tl.float64)
    pz = (line_ids % size).to(tl.float64)
    fya, fza = (ay + 1.0) / spacing, (az + 1.0) / spacing
    fyb, fzb = (by + 1.0) / spacing, (bz + 1.0) / spacing
    fyc, fzc = (cy + 1.0) / spacing, (cz + 1.0) / spacing
    fxa, fxb, fxc = (ax + 1.0) / spacing, (bx + 1.0) / spacing, (cx + 1.0) / spacing
    ya, za = fya - py, fza - pz
    yb, zb = fyb - py, fzb - pz
    yc, zc = fyc - py, fzc - pz
    weight_a = zb * yc - yb * zc
    weight_b = zc * ya - yc * za
    weight_c = za * yb - ya * zb
    sign_a = _orientation(yb, zb, yc, zc)
    sign_b = _orientation(yc, zc, ya, za)
    sign_c = _orientation(ya, za, yb, zb)
    total = weight_a + weight_b + weight_c
    inside = valid & (sign_a != 0) & (sign_b == sign_a) & (sign_c == sign_a)
    denominator = tl.where(inside, total, 1.0)
    intersection = (weight_a * fxa + weight_b * fxb + weight_c * fxc) / denominator
    interval = tl.ceil(intersection).to(tl.int32)
    interval = tl.maximum(interval, 0)
    tl.atomic_add(
        counts + interval * lines + line_ids,
        1,
        mask=inside & (interval < size),
    )


@triton.jit
def _sign_kernel(
    distances,
    counts,
    size: tl.constexpr,
    block_x: tl.constexpr,
    block_lines: tl.constexpr,
):
    line_ids = tl.program_id(0) * block_lines + tl.arange(0, block_lines)[None, :]
    ix = tl.arange(0, block_x)[:, None]
    lines = size * size
    offsets = ix * lines + line_ids
    valid = (ix < size) & (line_ids < lines)
    intersections = tl.load(counts + offsets, mask=valid, other=0)
    parity = tl.cumsum(intersections, axis=0) % 2
    distance = tl.load(distances + offsets, mask=valid, other=0.0)
    tl.store(
        distances + offsets, tl.where(parity == 1, -distance, distance), mask=valid
    )


def _launch_intersections(
    kernel: Callable[..., None],
    triangles: Tensor,
    counts: Tensor,
    n_triangles: int,
    size: int,
) -> None:
    kernel(triangles, counts, n_triangles, size, 256, enable_fp_fusion=False)


def _launch_sign(
    kernel: Callable[..., None],
    output: Tensor,
    counts: Tensor,
    size: int,
    block_x: int,
) -> None:
    kernel(output, counts, size, block_x, 16, enable_fp_fusion=False)


def apply_signs(triangles: Tensor, output: Tensor, size: int) -> Tensor:
    counts = torch.zeros(size**3, dtype=torch.int32, device=triangles.device)
    n_triangles = triangles.shape[0]
    _launch_intersections(
        _intersection_kernel[(triton.cdiv(n_triangles * size * size, 256),)],
        triangles,
        counts,
        n_triangles,
        size,
    )
    block_x = 1 << (size - 1).bit_length()
    _launch_sign(
        _sign_kernel[(triton.cdiv(size * size, 16),)], output, counts, size, block_x
    )
    return output.reshape(size, size, size)
