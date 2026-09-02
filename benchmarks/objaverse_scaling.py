# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-image", "torch", "trimesh", "triton"]
# ///
# ─── How to run ───
# From the repository root:
# /home/rvi/conda/envs/torch/bin/python -B benchmarks/objaverse_scaling.py verify

"""Objaverse accuracy and face-count scaling benchmark for autoresearch."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
import torch
import trimesh
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mesh2sdf
import mesh2sdf.core

OBJAVERSE_ROOT: Final = Path("/home/rvi/ns3/jaehyeok/ds/Objaverse-full/glbs/000-082")
ACCURACY_TIER: Final = 1000.0
MAX_ERROR: Final = 1e-3
MEAN_ERROR: Final = 1e-5
RELIABLE_DISTANCE: Final = 1e-4
FAILED_MEASUREMENT: Final = 1e9


@dataclass(frozen=True, slots=True)
class CaseSpec:
    uid: str
    expected_faces: int


CASES: Final = (
    CaseSpec("46643f0907de491289e4a294539efbe0", 236),
    CaseSpec("359bfd5cb9024ce5a14fc3a1cd6dfadb", 5_120),
    CaseSpec("946f31014a7c4595871c4874f670e215", 51_748),
    CaseSpec("8eda48c250214fd08017860a2a1520ef", 332_820),
)


class ResultRow(TypedDict):
    case: str
    faces: int
    size: int
    cpu_seconds: float
    gpu_seconds: float
    speedup: float
    max_error: float
    mean_error: float
    sign_errors: int
    accuracy_pass: bool
    error: str | None


def scalability_score(accuracy_passes: int, minimum_speedup: float) -> float:
    """Prioritize another accurate case over any speed improvement."""
    speed_tiebreaker = min(max(minimum_speedup, 0.0), ACCURACY_TIER - 1e-3)
    return accuracy_passes * ACCURACY_TIER + speed_tiebreaker


def accuracy_gated_speedup(speedup: float, *, accuracy_pass: bool) -> float:
    """Make an inaccurate dense result ineligible for performance retention."""
    return speedup if accuracy_pass else 0.0


def _load_case(spec: CaseSpec) -> tuple[NDArray[np.float32], NDArray[np.uint32]]:
    scene = trimesh.load_scene(
        OBJAVERSE_ROOT / f"{spec.uid}.glb", process=False, skip_materials=True
    )
    mesh = scene.to_mesh()
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    if not mesh.is_watertight:
        msg = f"{spec.uid} is no longer watertight"
        raise RuntimeError(msg)
    if len(mesh.faces) != spec.expected_faces:
        msg = f"{spec.uid} has {len(mesh.faces)} faces, expected {spec.expected_faces}"
        raise RuntimeError(msg)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center = (bounds[1] + bounds[0]) * 0.5
    extent = float(np.max(bounds[1] - bounds[0]))
    vertices = np.ascontiguousarray(
        (mesh.vertices - center) * (1.8 / extent), dtype=np.float32
    )
    faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
    return vertices, faces


def _measure(
    call: Callable[[], NDArray[np.float32]], samples: int
) -> tuple[float, NDArray[np.float32]]:
    durations: list[float] = []
    result = call()
    for _ in range(samples):
        start = time.perf_counter()
        result = call()
        durations.append(time.perf_counter() - start)
    return statistics.median(durations), result


def _compare(spec: CaseSpec, size: int, timed: bool) -> ResultRow:
    vertices, faces = _load_case(spec)
    start = time.perf_counter()
    reference = mesh2sdf.core.compute(vertices, faces, size)
    cpu_seconds = time.perf_counter() - start

    def candidate() -> NDArray[np.float32]:
        return np.asarray(
            mesh2sdf.compute(vertices, faces, size=size, backend="cuda"),
            dtype=np.float32,
        )

    if timed:
        samples = 3 if len(faces) < 10_000 else 1
        gpu_seconds, result = _measure(candidate, samples)
    else:
        start = time.perf_counter()
        result = candidate()
        gpu_seconds = time.perf_counter() - start

    error = np.abs(result - reference)
    reliable = np.abs(reference) > RELIABLE_DISTANCE
    sign_errors = int(
        np.count_nonzero((np.signbit(result) != np.signbit(reference)) & reliable)
    )
    max_error = float(error.max())
    mean_error = float(error.mean())
    return {
        "case": spec.uid,
        "faces": len(faces),
        "size": size,
        "cpu_seconds": cpu_seconds,
        "gpu_seconds": gpu_seconds,
        "speedup": cpu_seconds / gpu_seconds,
        "max_error": max_error,
        "mean_error": mean_error,
        "sign_errors": sign_errors,
        "accuracy_pass": bool(
            np.isfinite(result).all()
            and max_error <= MAX_ERROR
            and mean_error <= MEAN_ERROR
            and sign_errors == 0
        ),
        "error": None,
    }


def _failed_row(spec: CaseSpec, size: int, error: str) -> ResultRow:
    return {
        "case": spec.uid,
        "faces": spec.expected_faces,
        "size": size,
        "cpu_seconds": 0.0,
        "gpu_seconds": FAILED_MEASUREMENT,
        "speedup": 0.0,
        "max_error": FAILED_MEASUREMENT,
        "mean_error": FAILED_MEASUREMENT,
        "sign_errors": -1,
        "accuracy_pass": False,
        "error": error,
    }


def _run_isolated(spec: CaseSpec, size: int, timed: bool) -> ResultRow:
    command = (
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "case",
        spec.uid,
        str(size),
        "timed" if timed else "guard",
    )
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=600, check=False
        )
    except subprocess.TimeoutExpired:
        return _failed_row(spec, size, "case timed out after 600 seconds")
    if completed.returncode != 0:
        return _failed_row(spec, size, completed.stderr.strip())
    payload = json.loads(completed.stdout.splitlines()[-1])
    return ResultRow(**payload)


def _find_case(uid: str) -> CaseSpec:
    for spec in CASES:
        if spec.uid == uid:
            return spec
    msg = f"unknown case: {uid}"
    raise RuntimeError(msg)


def verify() -> int:
    rows = [_run_isolated(spec, 128, timed=True) for spec in CASES]
    minimum_speedup = min(row["speedup"] for row in rows)
    accuracy_passes = sum(row["accuracy_pass"] for row in rows)
    print(
        json.dumps(
            {
                "cases": rows,
                "accuracy_passes": accuracy_passes,
                "minimum_speedup": minimum_speedup,
                "dense_speedup": accuracy_gated_speedup(
                    rows[-1]["speedup"], accuracy_pass=rows[-1]["accuracy_pass"]
                ),
                "scalability_score": scalability_score(
                    accuracy_passes, minimum_speedup
                ),
            }
        )
    )
    return 0


def guard() -> int:
    checks = [(spec, 64) for spec in CASES]
    checks.extend((spec, 128) for spec in CASES[:-1])
    rows = [_run_isolated(spec, size, timed=False) for spec, size in checks]
    passed = all(row["accuracy_pass"] for row in rows)
    print(json.dumps({"guard": "passed" if passed else "failed", "cases": rows}))
    return 0 if passed else 1


def main(argv: Sequence[str]) -> int:
    arguments = tuple(argv[1:])
    if arguments == ("verify",):
        return verify()
    if arguments == ("guard",):
        return guard()
    if len(arguments) == 4 and arguments[0] == "case":
        _, uid, size, timing = arguments
        spec = _find_case(uid)
        try:
            row = _compare(spec, int(size), timed=timing == "timed")
        except torch.AcceleratorError as error:
            row = _failed_row(spec, int(size), str(error))
        print(json.dumps(row))
        return 0
    print("usage: objaverse_scaling.py {verify|guard}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
