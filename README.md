# Mesh2SDF-Triton

Mesh2SDF-Triton is a standalone, GPU-only library for generating signed
distance fields from watertight meshes. It uses PyTorch and Triton for the
distance initialization, fast sweeping, and sign pass. Its distribution name
is `mesh2sdf-triton`; its Python import is `mesh2sdf_triton`.

It is derived from [Mesh2SDF](https://github.com/wang-ps/mesh2sdf), but it does
not bundle the original C++ extension, pybind11 bindings, or a CPU fallback.
Use the original project when a CPU implementation is required.

## Performance

The table compares Mesh2SDF-Triton with the original Mesh2SDF C++ package on
four normalized watertight Objaverse meshes at `size=128`. Measurements use
PyTorch 2.11.0, Triton 3.6.0, CUDA 12.8, and an NVIDIA RTX 3090. GPU timings
are post-warmup end-to-end calls, including host/device transfer and excluding
Triton's first-use JIT compilation.

| Faces | Original C++ CPU | Triton GPU | Speedup | Max absolute error | Sign errors |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 236 | 8.020 s | 0.236 s | 34.04x | 1.79e-7 | 0 |
| 5,120 | 8.662 s | 0.238 s | 36.36x | 1.71e-5 | 0 |
| 51,748 | 8.463 s | 0.238 s | 35.61x | 1.19e-7 | 0 |
| 332,820 | 10.821 s | 0.248 s | 43.65x | 9.52e-4 | 0 |

The comparison accepts finite output, a maximum error at most `1e-3`, mean
error at most `1e-5`, and no reliable sign mismatches against the original
implementation. The original package is only a development-time reference:
it runs in a separate Python environment and is never a dependency of this
package.

## How it differs from Mesh2SDF

| Area | Mesh2SDF | Mesh2SDF-Triton |
| --- | --- | --- |
| Execution | Single-core C++ CPU | CUDA GPU through PyTorch and Triton |
| Installation | Builds a pybind11 extension | Installs Python/Triton sources only |
| GPU kernels | None | Triangle-local distance updates, fast sweep, and projected sign traversal |
| CPU fallback | Available | Not included |
| Python import | `mesh2sdf` | `mesh2sdf_triton` |

Triton compiles its kernels on their first use for a GPU configuration and
caches them. There is no project-specific CUDA extension, no `nvcc` step, and
no C++ compiler requirement during installation.

## Install

Mesh2SDF-Triton requires Linux, an NVIDIA GPU, a CUDA-capable PyTorch build,
and Triton. Install the PyTorch build appropriate for the system first, then
install this package:

```shell
pip install torch
pip install mesh2sdf-triton
```

To install from a checkout:

```shell
git clone https://github.com/Kitsunetic/mesh2sdf-triton.git
cd mesh2sdf-triton
pip install .
```

Neither command builds C++ code or requires pybind11.

## Usage

Vertices must be normalized to `[-1, 1]`.

```python
import mesh2sdf_triton

sdf = mesh2sdf_triton.compute(
    vertices,
    faces,
    size=128,
    device="cuda:0",
)
```

`compute` returns a NumPy `float32` array. CUDA availability is required; the
call raises an error when the selected device cannot execute the Triton path.
The `device` argument accepts PyTorch CUDA device strings. `fix=True` retains
the original mesh-repair workflow and can return the repaired mesh with
`return_mesh=True`.

## Implementation

Mesh2SDF-Triton preserves the SDFGen-style construction while moving the
face-dependent work onto the GPU:

1. Triangle bounds are derived from GPU-resident triangles using C++-compatible
   grid rounding.
2. A triangle-local narrow-band kernel initializes nearest distances with a
   packed atomic minimum that also records the nearest triangle.
3. Two passes of eight-direction fast sweeping fill the distance field.
4. A projected-triangle YZ traversal counts X intersections before applying
   parity signs.

The projected sign traversal avoids evaluating every face against every grid
line. For the 332,820-face benchmark above, it reduces the candidate set from
5.45 billion face/line pairs to 12,861 projected cells.

## Accuracy and reference benchmarks

The signed path is intended for watertight meshes. For non-watertight input,
use `fix=True` before relying on signs.

The repository's parity and Objaverse benchmarks can compare against an
independently installed original Mesh2SDF. Set
`MESH2SDF_REFERENCE_PYTHON` to the interpreter in that environment so the
reference runs in a subprocess and cannot become a package dependency:

```shell
MESH2SDF_REFERENCE_PYTHON=/path/to/original-mesh2sdf/bin/python \
  python -B benchmarks/objaverse_scaling.py verify
```

Set `OBJAVERSE_ROOT` to the directory containing the benchmark GLB files when
it differs from the script's default.

## Original project

Mesh2SDF-Triton is derived from Mesh2SDF by Peng-Shuai Wang. The original
project introduced the mesh repair and SDF workflow used here and is described
in the paper below.

## Citation

```
@article {Wang-Sig2022,
  title      = {Dual Octree Graph Networks for Learning Adaptive Volumetric
                Shape Representations},
  author     = {Wang, Peng-Shuai and Liu, Yang and Tong, Xin},
  journal    = {ACM Transactions on Graphics (SIGGRAPH)},
  volume     = {41},
  number     = {4},
  year       = {2022},
}
```
