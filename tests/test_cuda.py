import numpy as np
import pytest
import torch

import mesh2sdf
import mesh2sdf.core
from benchmarks.autoresearch import Case, fixed_cases


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("case", fixed_cases(), ids=lambda case: case.name)
def test_cuda_matches_reference_when_mesh_is_watertight(case: Case) -> None:
    # Given a watertight mesh and the original C++ field
    reference = mesh2sdf.core.compute(case.vertices, case.faces, 32)

    # When the same input uses the GPU backend
    result = mesh2sdf.compute(case.vertices, case.faces, size=32, backend="cuda")

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
