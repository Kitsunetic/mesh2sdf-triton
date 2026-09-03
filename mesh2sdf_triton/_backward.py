"""Triton kernels for the piecewise SDF vertex subgradient."""

from __future__ import annotations

from collections.abc import Callable

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _vertex_gradient_kernel(
    vertices,
    faces,
    closest,
    signed_distances,
    output_gradient,
    vertex_gradients,
    size: tl.constexpr,
    point_count: tl.constexpr,
    block: tl.constexpr,
):
    point_ids = tl.program_id(0) * block + tl.arange(0, block)
    valid = point_ids < point_count
    triangle_ids = tl.load(closest + point_ids, mask=valid, other=-1)
    has_triangle = valid & (triangle_ids >= 0)
    safe_triangle_ids = tl.maximum(triangle_ids, 0)
    face_base = safe_triangle_ids * 3
    first_id = tl.load(faces + face_base, mask=has_triangle, other=0)
    second_id = tl.load(faces + face_base + 1, mask=has_triangle, other=0)
    third_id = tl.load(faces + face_base + 2, mask=has_triangle, other=0)

    first_base = first_id * 3
    second_base = second_id * 3
    third_base = third_id * 3
    ax = tl.load(vertices + first_base, mask=has_triangle, other=0.0)
    ay = tl.load(vertices + first_base + 1, mask=has_triangle, other=0.0)
    az = tl.load(vertices + first_base + 2, mask=has_triangle, other=0.0)
    bx = tl.load(vertices + second_base, mask=has_triangle, other=0.0)
    by = tl.load(vertices + second_base + 1, mask=has_triangle, other=0.0)
    bz = tl.load(vertices + second_base + 2, mask=has_triangle, other=0.0)
    cx = tl.load(vertices + third_base, mask=has_triangle, other=0.0)
    cy = tl.load(vertices + third_base + 1, mask=has_triangle, other=0.0)
    cz = tl.load(vertices + third_base + 2, mask=has_triangle, other=0.0)

    xy_size = size * size
    i = point_ids // xy_size
    remainder = point_ids % xy_size
    j = remainder // size
    k = remainder % size
    spacing = 2.0 / size
    px = -1.0 + i.to(tl.float32) * spacing
    py = -1.0 + j.to(tl.float32) * spacing
    pz = -1.0 + k.to(tl.float32) * spacing

    acx, acy, acz = ax - cx, ay - cy, az - cz
    bcx, bcy, bcz = bx - cx, by - cy, bz - cz
    pcx, pcy, pcz = px - cx, py - cy, pz - cz
    ac_squared = acx * acx + acy * acy + acz * acz
    bc_squared = bcx * bcx + bcy * bcy + bcz * bcz
    ac_bc = acx * bcx + acy * bcy + acz * bcz
    inverse = 1.0 / tl.maximum(ac_squared * bc_squared - ac_bc * ac_bc, 1e-30)
    ac_point = acx * pcx + acy * pcy + acz * pcz
    bc_point = bcx * pcx + bcy * pcy + bcz * pcz
    first_weight = inverse * (bc_squared * ac_point - ac_bc * bc_point)
    second_weight = inverse * (ac_squared * bc_point - ac_bc * ac_point)
    third_weight = 1.0 - first_weight - second_weight
    face_qx = first_weight * ax + second_weight * bx + third_weight * cx
    face_qy = first_weight * ay + second_weight * by + third_weight * cy
    face_qz = first_weight * az + second_weight * bz + third_weight * cz

    ab_dx, ab_dy, ab_dz = bx - ax, by - ay, bz - az
    ab_length = ab_dx * ab_dx + ab_dy * ab_dy + ab_dz * ab_dz
    ab_weight = (
        (bx - px) * ab_dx + (by - py) * ab_dy + (bz - pz) * ab_dz
    ) / tl.maximum(ab_length, 1e-30)
    ab_weight = tl.minimum(tl.maximum(ab_weight, 0.0), 1.0)
    ab_qx, ab_qy, ab_qz = (
        ab_weight * ax + (1.0 - ab_weight) * bx,
        ab_weight * ay + (1.0 - ab_weight) * by,
        ab_weight * az + (1.0 - ab_weight) * bz,
    )
    ab_dx, ab_dy, ab_dz = px - ab_qx, py - ab_qy, pz - ab_qz
    ab_squared = ab_dx * ab_dx + ab_dy * ab_dy + ab_dz * ab_dz
    ac_dx, ac_dy, ac_dz = cx - ax, cy - ay, cz - az
    ac_length = ac_dx * ac_dx + ac_dy * ac_dy + ac_dz * ac_dz
    ac_weight = (
        (cx - px) * ac_dx + (cy - py) * ac_dy + (cz - pz) * ac_dz
    ) / tl.maximum(ac_length, 1e-30)
    ac_weight = tl.minimum(tl.maximum(ac_weight, 0.0), 1.0)
    ac_qx, ac_qy, ac_qz = (
        ac_weight * ax + (1.0 - ac_weight) * cx,
        ac_weight * ay + (1.0 - ac_weight) * cy,
        ac_weight * az + (1.0 - ac_weight) * cz,
    )
    ac_dx, ac_dy, ac_dz = px - ac_qx, py - ac_qy, pz - ac_qz
    ac_squared = ac_dx * ac_dx + ac_dy * ac_dy + ac_dz * ac_dz
    bc_dx, bc_dy, bc_dz = cx - bx, cy - by, cz - bz
    bc_length = bc_dx * bc_dx + bc_dy * bc_dy + bc_dz * bc_dz
    bc_weight = (
        (cx - px) * bc_dx + (cy - py) * bc_dy + (cz - pz) * bc_dz
    ) / tl.maximum(bc_length, 1e-30)
    bc_weight = tl.minimum(tl.maximum(bc_weight, 0.0), 1.0)
    bc_qx, bc_qy, bc_qz = (
        bc_weight * bx + (1.0 - bc_weight) * cx,
        bc_weight * by + (1.0 - bc_weight) * cy,
        bc_weight * bz + (1.0 - bc_weight) * cz,
    )
    bc_dx, bc_dy, bc_dz = px - bc_qx, py - bc_qy, pz - bc_qz
    bc_squared = bc_dx * bc_dx + bc_dy * bc_dy + bc_dz * bc_dz
    first_region = first_weight > 0.0
    second_region = second_weight > 0.0
    choose_ab = tl.where(
        first_region,
        ab_squared < ac_squared,
        tl.where(second_region, ab_squared < bc_squared, False),
    )
    choose_ac = tl.where(
        first_region,
        ab_squared >= ac_squared,
        tl.where(second_region, False, ac_squared < bc_squared),
    )
    edge_qx = tl.where(choose_ab, ab_qx, tl.where(choose_ac, ac_qx, bc_qx))
    edge_qy = tl.where(choose_ab, ab_qy, tl.where(choose_ac, ac_qy, bc_qy))
    edge_qz = tl.where(choose_ab, ab_qz, tl.where(choose_ac, ac_qz, bc_qz))
    edge_first_weight = tl.where(
        choose_ab, ab_weight, tl.where(choose_ac, ac_weight, 0.0)
    )
    edge_second_weight = tl.where(
        choose_ab, 1.0 - ab_weight, tl.where(choose_ac, 0.0, bc_weight)
    )
    edge_third_weight = tl.where(
        choose_ab, 0.0, tl.where(choose_ac, 1.0 - ac_weight, 1.0 - bc_weight)
    )
    inside = (first_weight >= 0.0) & (second_weight >= 0.0) & (third_weight >= 0.0)
    qx = tl.where(inside, face_qx, edge_qx)
    qy = tl.where(inside, face_qy, edge_qy)
    qz = tl.where(inside, face_qz, edge_qz)
    first_vertex_weight = tl.where(inside, first_weight, edge_first_weight)
    second_vertex_weight = tl.where(inside, second_weight, edge_second_weight)
    third_vertex_weight = tl.where(inside, third_weight, edge_third_weight)

    dx = qx - px
    dy = qy - py
    dz = qz - pz
    distance = tl.sqrt(tl.maximum(dx * dx + dy * dy + dz * dz, 0.0))
    signed_distance = tl.load(signed_distances + point_ids, mask=valid, other=0.0)
    upstream = tl.load(output_gradient + point_ids, mask=valid, other=0.0)
    sign = tl.where(signed_distance < 0.0, -1.0, 1.0)
    scale = tl.where(distance > 0.0, upstream * sign / distance, 0.0)
    gradient_x = scale * dx
    gradient_y = scale * dy
    gradient_z = scale * dz
    update = has_triangle & (distance > 0.0)

    tl.atomic_add(
        vertex_gradients + first_base, first_vertex_weight * gradient_x, mask=update
    )
    tl.atomic_add(
        vertex_gradients + first_base + 1, first_vertex_weight * gradient_y, mask=update
    )
    tl.atomic_add(
        vertex_gradients + first_base + 2, first_vertex_weight * gradient_z, mask=update
    )
    tl.atomic_add(
        vertex_gradients + second_base, second_vertex_weight * gradient_x, mask=update
    )
    tl.atomic_add(
        vertex_gradients + second_base + 1,
        second_vertex_weight * gradient_y,
        mask=update,
    )
    tl.atomic_add(
        vertex_gradients + second_base + 2,
        second_vertex_weight * gradient_z,
        mask=update,
    )
    tl.atomic_add(
        vertex_gradients + third_base, third_vertex_weight * gradient_x, mask=update
    )
    tl.atomic_add(
        vertex_gradients + third_base + 1, third_vertex_weight * gradient_y, mask=update
    )
    tl.atomic_add(
        vertex_gradients + third_base + 2, third_vertex_weight * gradient_z, mask=update
    )


def _launch(
    kernel: Callable[..., None],
    vertices: Tensor,
    faces: Tensor,
    closest: Tensor,
    signed_distances: Tensor,
    output_gradient: Tensor,
    vertex_gradients: Tensor,
    size: int,
    point_count: int,
) -> None:
    kernel(
        vertices,
        faces,
        closest,
        signed_distances,
        output_gradient,
        vertex_gradients,
        size,
        point_count,
        256,
    )


def accumulate_vertex_gradients(
    vertices: Tensor,
    faces: Tensor,
    closest: Tensor,
    signed_distances: Tensor,
    output_gradient: Tensor,
) -> Tensor:
    """Accumulate the fixed-feature SDF subgradient into vertex coordinates."""
    vertex_gradients = torch.zeros_like(vertices)
    point_count = signed_distances.numel()
    size = signed_distances.shape[0]
    kernel: Callable[..., None] = _vertex_gradient_kernel[
        (triton.cdiv(point_count, 256),)
    ]
    _launch(
        kernel,
        vertices,
        faces,
        closest,
        signed_distances.reshape(-1),
        output_gradient.reshape(-1),
        vertex_gradients,
        size,
        point_count,
    )
    return vertex_gradients
