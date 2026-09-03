"""Correctness tests for the native PyTorch SDF reference implementation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import mesh2sdf_triton
from benchmarks.autoresearch import Case, fixed_cases
from benchmarks.cpp_reference import REFERENCE_PYTHON_ENV, compute_reference

REFERENCE_AVAILABLE = Path(os.environ.get(REFERENCE_PYTHON_ENV, "")).is_file()
CASES: tuple[Case, ...] = fixed_cases()


@pytest.mark.skipif(
    not REFERENCE_AVAILABLE,
    reason="an independently installed original Mesh2SDF reference is required",
)
@pytest.mark.parametrize("case", CASES, ids=("box", "icosphere", "torus"))
@pytest.mark.parametrize("size", (8, 16))
def test_compute_pytorch_matches_original_mesh2sdf(case: Case, size: int) -> None:
    # Given a normalized watertight mesh and the original C++ Mesh2SDF field
    reference, _ = compute_reference(case.vertices, case.faces, size)
    vertices = torch.tensor(case.vertices.tolist(), dtype=torch.float32)
    faces = torch.tensor(case.faces.tolist(), dtype=torch.int64)

    # When the native PyTorch reference implementation computes its field
    result = mesh2sdf_triton.compute_pytorch(vertices, faces, size=size)

    # Then it reproduces the original construction to float32 precision
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert result.shape == (size, size, size)
    torch.testing.assert_close(
        result.detach().cpu(),
        torch.tensor(reference.tolist(), dtype=torch.float32),
        rtol=0.0,
        atol=1e-6,
    )
