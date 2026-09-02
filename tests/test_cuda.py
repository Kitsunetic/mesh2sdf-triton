import numpy as np
import pytest
import torch

import mesh2sdf
import mesh2sdf._cuda as cuda_backend
import mesh2sdf.core
from benchmarks.autoresearch import Case, fixed_cases
from mesh2sdf._distance import initialize_distance_grid


def test_projected_sign_bounds_use_ceil_floor_and_clamp() -> None:
    # Given a triangle whose projected YZ extent crosses both grid edges
    triangles = np.array(
        [[[-0.2, -0.74, -1.4], [0.1, -0.26, 0.1], [0.3, 0.24, 1.4]]],
        dtype=np.float64,
    )

    # When its sign-intersection bounds are computed in grid coordinates
    bounds = cuda_backend._projected_sign_bounds(triangles, spacing=0.25, size=8)

    # Then lower bounds use ceil, upper bounds use floor, and both are clamped
    np.testing.assert_array_equal(bounds, np.array([[2, 4, 0, 7]], dtype=np.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("case", fixed_cases(), ids=lambda case: case.name)
def test_cuda_matches_reference_when_mesh_is_watertight(case: Case) -> None:
    # Given a watertight mesh and the original C++ field
    reference = mesh2sdf.core.compute(case.vertices, case.faces, 64)

    # When the same input uses the GPU backend
    result = mesh2sdf.compute(case.vertices, case.faces, size=64, backend="cuda")

    # Then the public NumPy result preserves distances and reliable signs
    assert isinstance(result, np.ndarray)
    error = np.abs(result - reference)
    assert float(error.max()) <= 1e-3
    assert float(error.mean()) <= 1e-5
    reliable = np.abs(reference) > 1e-4
    assert np.array_equal(np.signbit(result[reliable]), np.signbit(reference[reliable]))


def test_cpu_backend_is_unchanged_when_explicitly_selected() -> None:
    # Given a fixed mesh and its C++ result
    case = fixed_cases()[0]
    reference = mesh2sdf.core.compute(case.vertices, case.faces, 16)

    # When CPU execution is explicitly requested
    result = mesh2sdf.compute(case.vertices, case.faces, size=16, backend="cpu")

    # Then the result is bitwise identical
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, reference)


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
