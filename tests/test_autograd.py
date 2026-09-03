"""Autograd contracts for the PyTorch oracle and Triton SDF path."""

from __future__ import annotations

import pytest
import torch

import mesh2sdf_triton
from benchmarks.autoresearch import fixed_cases


def test_compute_pytorch_assigns_zero_gradient_at_zero_distance() -> None:
    # Given a triangle that contains a grid vertex exactly on its surface
    vertices = torch.tensor(
        [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)

    # When the zero-distance sample is differentiated with respect to vertices
    distance = mesh2sdf_triton.compute_pytorch(vertices, faces, size=8)[2, 2, 4]
    gradient = torch.autograd.grad(distance, vertices)[0]

    # Then the selected zero subgradient is finite and exactly zero
    torch.testing.assert_close(gradient, torch.zeros_like(vertices), rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compute_triton_vertex_gradient_matches_pytorch_oracle() -> None:
    # Given a rotated watertight mesh away from closest-feature boundaries
    case = fixed_cases()[0]
    faces = torch.tensor(case.faces.tolist(), dtype=torch.int64, device="cuda")
    weights = torch.linspace(
        -0.4, 0.6, 8**3, dtype=torch.float32, device="cuda"
    ).reshape(8, 8, 8)
    pytorch_vertices = torch.tensor(
        case.vertices.tolist(), dtype=torch.float32, device="cuda", requires_grad=True
    )
    triton_vertices = pytorch_vertices.detach().clone().requires_grad_(True)

    # When both implementations differentiate the same weighted SDF loss
    pytorch_field = mesh2sdf_triton.compute_pytorch(pytorch_vertices, faces, size=8)
    triton_field = mesh2sdf_triton.compute_triton(triton_vertices, faces, size=8)
    pytorch_gradient = torch.autograd.grad(
        (pytorch_field * weights).sum(), pytorch_vertices
    )[0]
    triton_gradient = torch.autograd.grad(
        (triton_field * weights).sum(), triton_vertices
    )[0]

    # Then the CUDA custom backward follows the PyTorch subgradient convention
    torch.testing.assert_close(triton_field, pytorch_field, rtol=0.0, atol=1e-6)
    assert bool(torch.isfinite(pytorch_gradient).all())
    assert bool(torch.isfinite(triton_gradient).all())
    torch.testing.assert_close(triton_gradient, pytorch_gradient, rtol=1e-4, atol=1e-5)
