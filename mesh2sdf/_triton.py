"""Triton kernels for dense signed-distance grids."""

from __future__ import annotations

from collections.abc import Callable

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _segment_distance_squared(px, py, pz, ax, ay, az, bx, by, bz):
    dx = bx - ax
    dy = by - ay
    dz = bz - az
    length_squared = dx * dx + dy * dy + dz * dz
    numerator = (bx - px) * dx + (by - py) * dy + (bz - pz) * dz
    s = (numerator.to(tl.float64) /
         tl.maximum(length_squared.to(tl.float64), 1e-30)).to(tl.float32)
    s = tl.minimum(tl.maximum(s, 0.0), 1.0)
    qx = s * ax + (1.0 - s) * bx
    qy = s * ay + (1.0 - s) * by
    qz = s * az + (1.0 - s) * bz
    ex = px - qx
    ey = py - qy
    ez = pz - qz
    return ex * ex + ey * ey + ez * ez


@triton.jit
def _distance_kernel(
    triangles,
    bounds,
    output,
    closest,
    n_triangles: tl.constexpr,
    size: tl.constexpr,
    block_points: tl.constexpr,
    block_triangles: tl.constexpr,
):
    point_ids = tl.program_id(0) * block_points + tl.arange(0, block_points)
    valid_points = point_ids < size * size * size
    plane = size * size
    ix = point_ids // plane
    remainder = point_ids - ix * plane
    iy = remainder // size
    iz = remainder - iy * size
    spacing = 2.0 / size
    px = (-1.0 + ix.to(tl.float32) * spacing)[:, None]
    py = (-1.0 + iy.to(tl.float32) * spacing)[:, None]
    pz = (-1.0 + iz.to(tl.float32) * spacing)[:, None]
    best = tl.full((block_points,), 36.0, tl.float32)
    best_triangle = tl.full((block_points,), -1, tl.int32)

    for tile in range(tl.cdiv(n_triangles, block_triangles)):
        triangle_ids = tile * block_triangles + tl.arange(0, block_triangles)
        valid_triangles = triangle_ids < n_triangles
        base = triangle_ids[None, :] * 9
        bound_base = triangle_ids[None, :] * 6
        x0 = tl.load(bounds + bound_base, mask=valid_triangles[None, :], other=0)
        x1 = tl.load(bounds + bound_base + 1, mask=valid_triangles[None, :], other=-1)
        y0 = tl.load(bounds + bound_base + 2, mask=valid_triangles[None, :], other=0)
        y1 = tl.load(bounds + bound_base + 3, mask=valid_triangles[None, :], other=-1)
        z0 = tl.load(bounds + bound_base + 4, mask=valid_triangles[None, :], other=0)
        z1 = tl.load(bounds + bound_base + 5, mask=valid_triangles[None, :], other=-1)
        ax = tl.load(triangles + base, mask=valid_triangles[None, :], other=0.0)
        ay = tl.load(triangles + base + 1, mask=valid_triangles[None, :], other=0.0)
        az = tl.load(triangles + base + 2, mask=valid_triangles[None, :], other=0.0)
        bx = tl.load(triangles + base + 3, mask=valid_triangles[None, :], other=0.0)
        by = tl.load(triangles + base + 4, mask=valid_triangles[None, :], other=0.0)
        bz = tl.load(triangles + base + 5, mask=valid_triangles[None, :], other=0.0)
        cx = tl.load(triangles + base + 6, mask=valid_triangles[None, :], other=0.0)
        cy = tl.load(triangles + base + 7, mask=valid_triangles[None, :], other=0.0)
        cz = tl.load(triangles + base + 8, mask=valid_triangles[None, :], other=0.0)

        x13x, x13y, x13z = ax - cx, ay - cy, az - cz
        x23x, x23y, x23z = bx - cx, by - cy, bz - cz
        x03x, x03y, x03z = px - cx, py - cy, pz - cz
        m13 = x13x * x13x + x13y * x13y + x13z * x13z
        m23 = x23x * x23x + x23y * x23y + x23z * x23z
        dot_13_23 = x13x * x23x + x13y * x23y + x13z * x23z
        inverse = 1.0 / tl.maximum(m13 * m23 - dot_13_23 * dot_13_23, 1e-30)
        dot_13_03 = x13x * x03x + x13y * x03y + x13z * x03z
        dot_23_03 = x23x * x03x + x23y * x03y + x23z * x03z
        w23 = inverse * (m23 * dot_13_03 - dot_13_23 * dot_23_03)
        w31 = inverse * (m13 * dot_23_03 - dot_13_23 * dot_13_03)
        w12 = 1.0 - w23 - w31
        qx = w23 * ax + w31 * bx + w12 * cx
        qy = w23 * ay + w31 * by + w12 * cy
        qz = w23 * az + w31 * bz + w12 * cz
        plane_dx, plane_dy, plane_dz = px - qx, py - qy, pz - qz
        plane_distance = (
            plane_dx * plane_dx + plane_dy * plane_dy + plane_dz * plane_dz
        )
        distance_ab = _segment_distance_squared(px, py, pz, ax, ay, az, bx, by, bz)
        distance_ac = _segment_distance_squared(px, py, pz, ax, ay, az, cx, cy, cz)
        distance_bc = _segment_distance_squared(px, py, pz, bx, by, bz, cx, cy, cz)
        edge_distance = tl.where(
            w23 > 0.0,
            tl.minimum(distance_ab, distance_ac),
            tl.where(w31 > 0.0, tl.minimum(distance_ab, distance_bc),
                     tl.minimum(distance_ac, distance_bc)),
        )
        inside = (w23 >= 0.0) & (w31 >= 0.0) & (w12 >= 0.0)
        distance = tl.where(inside, plane_distance, edge_distance)
        in_band = (
            valid_triangles[None, :]
            & (ix[:, None] >= x0) & (ix[:, None] <= x1)
            & (iy[:, None] >= y0) & (iy[:, None] <= y1)
            & (iz[:, None] >= z0) & (iz[:, None] <= z1)
        )
        distance = tl.where(in_band, distance, float("inf"))
        tile_best = tl.min(distance, axis=1)
        tile_argmin = tl.argmin(distance, axis=1, tie_break_left=True)
        update = tile_best < best
        best = tl.where(update, tile_best, best)
        best_triangle = tl.where(update, tile * block_triangles + tile_argmin,
                                 best_triangle)

    tl.store(output + point_ids, tl.sqrt(tl.maximum(best, 0.0)), mask=valid_points)
    tl.store(closest + point_ids, best_triangle, mask=valid_points)


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
                tl.where(y2 < y1, -1,
                         tl.where(x1 > x2, 1, tl.where(x1 < x2, -1, 0))),
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
    intersection = (
        weight_a * fxa + weight_b * fxb + weight_c * fxc
    ) / denominator
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
    tl.store(distances + offsets, tl.where(parity == 1, -distance, distance), mask=valid)


def _launch_distance(
    kernel: Callable[..., None], triangles: Tensor, bounds: Tensor,
    output: Tensor, closest: Tensor, n_triangles: int, size: int,
) -> None:
    kernel(triangles, bounds, output, closest, n_triangles, size, 16, 32,
           enable_fp_fusion=False)


def _launch_intersections(
    kernel: Callable[..., None], triangles: Tensor, counts: Tensor,
    n_triangles: int, size: int,
) -> None:
    kernel(triangles, counts, n_triangles, size, 256, enable_fp_fusion=False)


def _launch_sign(
    kernel: Callable[..., None], output: Tensor, counts: Tensor,
    size: int, block_x: int,
) -> None:
    kernel(output, counts, size, block_x, 16, enable_fp_fusion=False)


def initialize_distance_grid(
    triangles: Tensor, bounds: Tensor, size: int,
) -> tuple[Tensor, Tensor]:
    output = torch.empty(size**3, dtype=torch.float32, device=triangles.device)
    closest = torch.empty(size**3, dtype=torch.int32, device=triangles.device)
    n_triangles = triangles.shape[0]
    _launch_distance(_distance_kernel[(triton.cdiv(size**3, 16),)],
                     triangles, bounds, output, closest, n_triangles, size)
    return output, closest


def apply_signs(triangles: Tensor, output: Tensor, size: int) -> Tensor:
    counts = torch.zeros(size**3, dtype=torch.int32, device=triangles.device)
    n_triangles = triangles.shape[0]
    _launch_intersections(
        _intersection_kernel[(triton.cdiv(n_triangles * size * size, 256),)],
        triangles, counts, n_triangles, size,
    )
    block_x = 1 << (size - 1).bit_length()
    _launch_sign(_sign_kernel[(triton.cdiv(size * size, 16),)],
                 output, counts, size, block_x)
    return output.reshape(size, size, size)
