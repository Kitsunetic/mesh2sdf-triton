import inspect

import numpy as np
import pytest
import torch

import mesh2sdf_triton
import mesh2sdf_triton._cuda as cuda_backend
from benchmarks.autoresearch import Case, fixed_cases
from mesh2sdf_triton._distance import initialize_distance_grid


def test_triangle_bounds_preserve_grid_rounding_and_clamping() -> None:
    # Given float32 triangle coordinates whose YZ extent crosses both grid edges
    triangles = torch.tensor(
        [[[-0.5, -0.75, -1.5], [0.0, -0.25, 0.0], [0.5, 0.25, 1.5]]],
        dtype=torch.float32,
    )

    # When distance and sign bounds are computed in grid coordinates
    distance, sign = cuda_backend._triangle_bounds(triangles, size=8)

    # Then the expected truncation, ceil, floor, and clamping rules are preserved
    torch.testing.assert_close(
        distance, torch.tensor([[1, 7, 0, 7, 0, 7]], dtype=torch.int32)
    )
    torch.testing.assert_close(sign, torch.tensor([[1, 5, 0, 7]], dtype=torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("case", fixed_cases(), ids=lambda case: case.name)
@pytest.mark.parametrize("size", (8, 16, 32, 64))
def test_public_cuda_api_runs_for_watertight_meshes(case: Case, size: int) -> None:
    # Given a watertight mesh and a requested CUDA grid size

    # When the standalone public API computes its SDF
    result = mesh2sdf_triton.compute(case.vertices, case.faces, size=size)

    # Then Triton returns a finite float32 field at the requested resolution
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == (size, size, size)
    assert np.isfinite(result).all()


def test_public_api_exposes_no_cpu_backend() -> None:
    parameters = inspect.signature(mesh2sdf_triton.compute).parameters

    assert "backend" not in parameters
    assert parameters["device"].default == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_distance_initialization_keeps_first_triangle_on_exact_tie() -> None:
    # Given two identical triangles with one shared narrow-band AABB
    triangle = torch.tensor(
        [[[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]]],
        dtype=torch.float32,
        device="cuda",
    )
    triangles = triangle.repeat(2, 1, 1)
    bounds = torch.tensor(
        [[1, 6, 1, 6, 3, 5], [1, 6, 1, 6, 3, 5]],
        dtype=torch.int32,
        device="cuda",
    )

    # When the narrow-band distances are initialized
    distances, closest = initialize_distance_grid(triangles, bounds, size=8)

    # Then exact ties choose the first triangle and untouched cells keep sentinels
    assert int(closest.reshape(8, 8, 8)[4, 4, 4]) == 0
    assert int(closest.reshape(8, 8, 8)[0, 0, 0]) == -1
    assert float(distances.reshape(8, 8, 8)[0, 0, 0]) == 6.0
