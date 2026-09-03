"""Fixed end-to-end timing and optional C++ parity checks for Mesh2SDF-Triton."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import trimesh
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mesh2sdf_triton
from benchmarks.cpp_reference import compute_reference


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    vertices: NDArray[np.float32]
    faces: NDArray[np.uint32]
    is_watertight: bool


def fixed_cases() -> tuple[Case, ...]:
    box = trimesh.creation.box(extents=(1.25, 1.0, 0.8))
    box.apply_transform(trimesh.transformations.rotation_matrix(0.37, (1, 2, 3)))
    meshes = (
        ("box", box),
        ("icosphere", trimesh.creation.icosphere(subdivisions=3, radius=0.8)),
        (
            "torus",
            trimesh.creation.torus(
                major_radius=0.55,
                minor_radius=0.22,
                major_sections=48,
                minor_sections=24,
            ),
        ),
    )
    return tuple(
        Case(
            name,
            np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.uint32),
            bool(mesh.is_watertight),
        )
        for name, mesh in meshes
    )


def measured_seconds(call: Callable[[], NDArray[np.float32]]) -> float:
    samples: list[float] = []
    for _ in range(3):
        start = time.perf_counter()
        result = call()
        samples.append(time.perf_counter() - start)
        if not np.isfinite(result).all():
            raise FloatingPointError("SDF contains non-finite values")
    return statistics.median(samples)


def measured_reference_seconds(
    call: Callable[[], tuple[NDArray[np.float32], float]]
) -> float:
    samples: list[float] = []
    for _ in range(3):
        result, seconds = call()
        if not np.isfinite(result).all():
            raise FloatingPointError("SDF contains non-finite values")
        samples.append(seconds)
    return statistics.median(samples)


def compute_candidate(case: Case, size: int) -> NDArray[np.float32]:
    return np.asarray(
        mesh2sdf_triton.compute(case.vertices, case.faces, size=size), dtype=np.float32
    )


def verify() -> None:
    rows: list[dict[str, str | float | int]] = []
    for case in fixed_cases():
        reference = partial(compute_reference, case.vertices, case.faces, 128)
        candidate = partial(compute_candidate, case, 128)
        candidate()  # Exclude the first JIT compilation, but no mesh preprocessing.
        reference_seconds = measured_reference_seconds(reference)
        candidate_seconds = measured_seconds(candidate)
        rows.append(
            {
                "case": case.name,
                "faces": len(case.faces),
                "reference_seconds": reference_seconds,
                "candidate_seconds": candidate_seconds,
                "speedup": reference_seconds / candidate_seconds,
            }
        )
    speedups = [float(row["speedup"]) for row in rows]
    print(json.dumps({"cases": rows, "min_speedup": min(speedups)}))


def guard() -> None:
    rows: list[dict[str, str | float | int]] = []
    for case in fixed_cases():
        for size in (32, 64):
            reference, _ = compute_reference(case.vertices, case.faces, size)
            candidate = compute_candidate(case, size)
            assert candidate.shape == reference.shape
            assert candidate.dtype == np.float32
            assert np.isfinite(candidate).all()
            error = np.abs(candidate - reference)
            max_error = float(error.max())
            mean_error = float(error.mean())
            reliable = np.abs(reference) > 1e-4
            sign_errors = int(
                np.count_nonzero(
                    (np.signbit(candidate) != np.signbit(reference)) & reliable
                )
            )
            row = {
                "case": case.name,
                "size": size,
                "max_error": max_error,
                "mean_error": mean_error,
                "sign_errors": sign_errors,
            }
            rows.append(row)
            assert max_error <= 1e-3, row
            assert mean_error <= 1e-5, row
            assert sign_errors == 0, row
    print(json.dumps({"guard": "passed", "cases": rows}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "guard"))
    args = parser.parse_args()
    if args.mode == "verify":
        verify()
    else:
        guard()


if __name__ == "__main__":
    main()
