"""Triangle-local narrow-band distance initialization kernels."""

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
    s = (
        numerator.to(tl.float64) / tl.maximum(length_squared.to(tl.float64), 1e-30)
    ).to(tl.float32)
    s = tl.minimum(tl.maximum(s, 0.0), 1.0)
    qx = s * ax + (1.0 - s) * bx
    qy = s * ay + (1.0 - s) * by
    qz = s * az + (1.0 - s) * bz
    ex = px - qx
    ey = py - qy
    ez = pz - qz
    return ex * ex + ey * ey + ez * ez


@triton.jit
def _triangle_distance_kernel(
    triangles,
    bounds,
    packed_closest,
    size: tl.constexpr,
    block_cells: tl.constexpr,
):
    triangle_id = tl.program_id(0)
    bound_base = triangle_id * 6
    x0 = tl.load(bounds + bound_base)
    x1 = tl.load(bounds + bound_base + 1)
    y0 = tl.load(bounds + bound_base + 2)
    y1 = tl.load(bounds + bound_base + 3)
    z0 = tl.load(bounds + bound_base + 4)
    z1 = tl.load(bounds + bound_base + 5)
    width_y = y1 - y0 + 1
    width_z = z1 - z0 + 1
    cell_count = (x1 - x0 + 1) * width_y * width_z

    base = triangle_id * 9
    ax = tl.load(triangles + base)
    ay = tl.load(triangles + base + 1)
    az = tl.load(triangles + base + 2)
    bx = tl.load(triangles + base + 3)
    by = tl.load(triangles + base + 4)
    bz = tl.load(triangles + base + 5)
    cx = tl.load(triangles + base + 6)
    cy = tl.load(triangles + base + 7)
    cz = tl.load(triangles + base + 8)
    spacing = 2.0 / size
    cell_base = 0
    while cell_base < cell_count:
        cell_ids = cell_base + tl.arange(0, block_cells)
        valid = cell_ids < cell_count
        ix = x0 + cell_ids // (width_y * width_z)
        remainder = cell_ids % (width_y * width_z)
        iy = y0 + remainder // width_z
        iz = z0 + remainder % width_z
        px = -1.0 + ix.to(tl.float32) * spacing
        py = -1.0 + iy.to(tl.float32) * spacing
        pz = -1.0 + iz.to(tl.float32) * spacing
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
        plane_distance = plane_dx * plane_dx + plane_dy * plane_dy + plane_dz * plane_dz
        distance_ab = _segment_distance_squared(px, py, pz, ax, ay, az, bx, by, bz)
        distance_ac = _segment_distance_squared(px, py, pz, ax, ay, az, cx, cy, cz)
        distance_bc = _segment_distance_squared(px, py, pz, bx, by, bz, cx, cy, cz)
        edge_distance = tl.where(
            w23 > 0.0,
            tl.minimum(distance_ab, distance_ac),
            tl.where(
                w31 > 0.0,
                tl.minimum(distance_ab, distance_bc),
                tl.minimum(distance_ac, distance_bc),
            ),
        )
        inside = (w23 >= 0.0) & (w31 >= 0.0) & (w12 >= 0.0)
        distance = tl.where(inside, plane_distance, edge_distance)
        distance_bits = distance.to(tl.int32, bitcast=True).to(tl.int64)
        candidate = (distance_bits << 32) | triangle_id
        point_ids = ix * size * size + iy * size + iz
        tl.atomic_min(
            packed_closest + point_ids,
            candidate,
            mask=valid & (distance < 36.0),
        )
        cell_base += block_cells


@triton.jit
def _unpack_distance_kernel(
    packed_closest,
    output,
    closest,
    point_count: tl.constexpr,
    block_points: tl.constexpr,
):
    point_ids = tl.program_id(0) * block_points + tl.arange(0, block_points)
    valid = point_ids < point_count
    packed = tl.load(packed_closest + point_ids, mask=valid, other=0)
    untouched = packed == 0x7FFFFFFFFFFFFFFF
    distance_bits = (packed >> 32).to(tl.int32)
    distance_squared = distance_bits.to(tl.float32, bitcast=True)
    distance_squared = tl.where(untouched, 36.0, distance_squared)
    triangle_id = tl.where(untouched, -1, packed.to(tl.int32))
    tl.store(output + point_ids, tl.sqrt(tl.maximum(distance_squared, 0.0)), mask=valid)
    tl.store(closest + point_ids, triangle_id, mask=valid)


def _launch_triangle_distances(
    kernel: Callable[..., None],
    triangles: Tensor,
    bounds: Tensor,
    packed_closest: Tensor,
    size: int,
) -> None:
    kernel(triangles, bounds, packed_closest, size, 256, enable_fp_fusion=False)


def _launch_unpack_distances(
    kernel: Callable[..., None],
    packed_closest: Tensor,
    output: Tensor,
    closest: Tensor,
    point_count: int,
) -> None:
    kernel(packed_closest, output, closest, point_count, 256, enable_fp_fusion=False)


def initialize_distance_grid(
    triangles: Tensor,
    bounds: Tensor,
    size: int,
) -> tuple[Tensor, Tensor]:
    point_count = size**3
    output = torch.empty(point_count, dtype=torch.float32, device=triangles.device)
    closest = torch.empty(point_count, dtype=torch.int32, device=triangles.device)
    packed_closest = torch.full(
        (point_count,),
        torch.iinfo(torch.int64).max,
        dtype=torch.int64,
        device=triangles.device,
    )
    n_triangles = triangles.shape[0]
    _launch_triangle_distances(
        _triangle_distance_kernel[(n_triangles,)],
        triangles,
        bounds,
        packed_closest,
        size,
    )
    _launch_unpack_distances(
        _unpack_distance_kernel[(triton.cdiv(point_count, 256),)],
        packed_closest,
        output,
        closest,
        point_count,
    )
    return output, closest
