"""Generate a Mesh2SDF-Triton field from a mesh file."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import cast

import numpy as np
import trimesh
from numpy.typing import NDArray

import mesh2sdf_triton


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input mesh file")
    parser.add_argument("--output", type=Path, help="output .npy path")
    parser.add_argument("--size", type=int, default=128, help="SDF resolution")
    parser.add_argument("--device", default="cuda", help="CUDA device")
    parser.add_argument("--no-fix", dest="fix", action="store_false")
    parser.set_defaults(fix=True)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    input_path = arguments.input
    output_path = arguments.output or input_path.with_suffix(".npy")
    mesh = cast(trimesh.Trimesh, trimesh.load(input_path, force="mesh"))

    bounds = np.asarray(mesh.bounds, dtype=np.float32)
    center = (bounds[0] + bounds[1]) * 0.5
    scale = 1.6 / float(np.max(bounds[1] - bounds[0]))
    vertices = np.asarray((mesh.vertices - center) * scale, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)

    started = time.perf_counter()
    if arguments.fix:
        repaired_result = mesh2sdf_triton.compute(
            vertices, faces, size=arguments.size, level=2.0 / arguments.size,
            return_mesh=True, device=arguments.device,
        )
        sdf, repaired_mesh = cast(
            tuple[NDArray[np.float32], trimesh.Trimesh], repaired_result
        )
        repaired_mesh.vertices = repaired_mesh.vertices / scale + center
        repaired_mesh.export(output_path.with_suffix(".fixed.obj"))
    else:
        result = mesh2sdf_triton.compute(
            vertices, faces, size=arguments.size, device=arguments.device,
        )
        sdf = cast(NDArray[np.float32], result)
    np.save(output_path, sdf)
    elapsed = time.perf_counter() - started
    print(f"Wrote {output_path} in {elapsed:.4f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
