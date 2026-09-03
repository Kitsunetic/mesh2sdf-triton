"""Optional subprocess adapter for an independently installed Mesh2SDF reference."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
from numpy.typing import NDArray

REFERENCE_PYTHON_ENV: Final = "MESH2SDF_REFERENCE_PYTHON"
_REFERENCE_PROGRAM: Final = """
import json
import sys
import time

import mesh2sdf
import numpy as np

input_path, output_path = sys.argv[1:]
payload = np.load(input_path)
started = time.perf_counter()
result = mesh2sdf.compute(payload["vertices"], payload["faces"], int(payload["size"]))
seconds = time.perf_counter() - started
np.save(output_path, result)
print(json.dumps({"seconds": seconds}))
"""


class _ReferenceMetadata(TypedDict):
  seconds: float


class ReferenceUnavailableError(RuntimeError):
  """Raised when the external original Mesh2SDF interpreter is unavailable."""


def _reference_python() -> Path:
  configured = os.environ.get(REFERENCE_PYTHON_ENV)
  if configured is None:
    message = (
        f"Set {REFERENCE_PYTHON_ENV} to a Python executable with the original "
        "mesh2sdf package installed."
    )
    raise ReferenceUnavailableError(message)
  executable = Path(configured)
  if not executable.is_file():
    message = f"{REFERENCE_PYTHON_ENV} does not name a file: {executable}"
    raise ReferenceUnavailableError(message)
  return executable


def compute_reference(
    vertices: NDArray[np.generic], faces: NDArray[np.generic], size: int
) -> tuple[NDArray[np.float32], float]:
  """Compute an original Mesh2SDF field without importing it into this process."""
  executable = _reference_python()
  environment = os.environ.copy()
  _ = environment.pop("PYTHONPATH", None)
  with tempfile.TemporaryDirectory(prefix="mesh2sdf-reference-") as directory:
    input_path = Path(directory) / "input.npz"
    output_path = Path(directory) / "output.npy"
    np.savez(
        input_path,
        vertices=np.ascontiguousarray(vertices, dtype=np.float32),
        faces=np.ascontiguousarray(faces, dtype=np.uint32),
        size=np.asarray(size, dtype=np.int64),
    )
    completed = subprocess.run(
        (
            str(executable),
            "-B",
            "-c",
            _REFERENCE_PROGRAM,
            str(input_path),
            str(output_path),
        ),
        capture_output=True,
        check=False,
        cwd=directory,
        env=environment,
        text=True,
    )
    if completed.returncode != 0:
      message = completed.stderr.strip() or "reference interpreter failed"
      raise ReferenceUnavailableError(message)
    raw_metadata: object = json.loads(completed.stdout)
    if not isinstance(raw_metadata, dict):
      raise ReferenceUnavailableError("reference interpreter returned invalid metadata")
    seconds = raw_metadata.get("seconds")
    if not isinstance(seconds, (int, float)):
      raise ReferenceUnavailableError("reference interpreter did not report elapsed time")
    metadata: _ReferenceMetadata = {"seconds": float(seconds)}
    result = np.asarray(np.load(output_path), dtype=np.float32)
    return result, float(metadata["seconds"])
