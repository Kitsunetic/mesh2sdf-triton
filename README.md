# Mesh2SDF-Triton

Mesh2SDF-Triton is a GPU-accelerated fork of
[Mesh2SDF](https://github.com/wang-ps/mesh2sdf). It keeps the original public
API and C++ backend, and adds a PyTorch + Triton backend for fast signed
distance-field generation from watertight meshes.

The original project is a robust CPU implementation of SDFGen-style sweeping.
This fork targets the preprocessing use case where many `size=128` grids must
be generated from large meshes on a CUDA machine.


## What changes from Mesh2SDF?

| Area | Original Mesh2SDF | Mesh2SDF-Triton |
| --- | --- | --- |
| Accelerated backend | C++ CPU | PyTorch + Triton CUDA kernels |
| Distance initialization | CPU triangle/voxel loops | GPU triangle-local narrow-band updates |
| Sign computation | CPU ray-intersection counting | GPU projected-triangle intersection counting |
| Large-mesh scaling | Work increases with face count on one CPU core | Face-dependent work stays on the GPU and is bounded by triangle grid coverage |
| Compatibility | Original behavior | `backend="cpu"` preserves the original C++ path |

The CUDA backend needs no custom CUDA extension and no `nvcc` build. Triton
JIT-compiles kernels when they are first used for a GPU configuration, then
reuses them on later calls.


## Performance

The table below compares the original C++ backend with the CUDA backend on
four normalized watertight Objaverse meshes at `size=128`. Measurements use
PyTorch 2.11.0, Triton 3.6.0, CUDA 12.8, and an NVIDIA RTX 3090. CUDA timings
are post-warmup end-to-end calls; they include host/device data movement and
exclude Triton's one-time compilation cost.

| Faces | C++ CPU | CUDA GPU | Speedup | Max absolute error | Sign errors |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 236 | 8.020 s | 0.236 s | 34.04x | 1.79e-7 | 0 |
| 5,120 | 8.662 s | 0.238 s | 36.36x | 1.71e-5 | 0 |
| 51,748 | 8.463 s | 0.238 s | 35.61x | 1.19e-7 | 0 |
| 332,820 | 10.821 s | 0.248 s | 43.65x | 9.52e-4 | 0 |

The benchmark checks finite outputs, a maximum error no greater than `1e-3`,
mean error no greater than `1e-5`, and zero sign mismatches against the C++
backend. Reproduce it with:

```shell
python -B benchmarks/objaverse_scaling.py verify
```

Set `OBJAVERSE_ROOT` in that script to a local directory containing the
benchmark GLB files before running it.


## Install

`mesh2sdf` depends on [pybind11](https://github.com/pybind/pybind11), and C++
compilers are needed to build the code. Supported compilers are listed
[here](https://github.com/pybind/pybind11#supported-compilers).

Clone this repository because the upstream PyPI package does not include the
Triton backend:

```shell
git clone https://github.com/Kitsunetic/mesh2sdf-triton.git
cd mesh2sdf-triton
pip install ".[cuda]"
```

The package still builds the original pybind11 C++ extension, so a supported C++
compiler is required. CUDA acceleration requires Linux, a CUDA-capable PyTorch
installation, and Triton.


## Use the accelerated backend

Vertices must be normalized to `[-1, 1]`, as in the original project. For grids
of size 64 or larger, `backend="auto"` selects CUDA when PyTorch, Triton, and a
CUDA device are available.

```python
import mesh2sdf

sdf = mesh2sdf.compute(
    vertices,
    faces,
    size=128,
    backend="cuda",
    device="cuda:0",
)
```

Use `backend="cpu"` for the original C++ implementation. `backend="cuda"`
raises an error if CUDA acceleration is unavailable, which is useful in batch
preprocessing jobs that must not silently fall back to CPU.


## Why large meshes stay fast

Mesh2SDF-Triton follows the same SDFGen-style result construction as the C++
backend, but changes where the expensive work happens.

1. Triangle bounds are computed from the already-uploaded triangle tensor on
   the GPU. Float64 arithmetic preserves the C++ grid rounding rules.
2. Distance initialization visits only the inclusive AABB around each triangle,
   rather than evaluating every triangle at every grid voxel. A packed atomic
   minimum selects both the nearest distance and its triangle consistently.
3. The original eight-direction fast sweep fills the remaining grid values in
   the same dependency order as the C++ implementation.
4. The sign pass visits only grid lines within a triangle's YZ projection. This
   avoids launching `face_count × size²` intersection tests; the 332,820-face
   benchmark reduces that candidate set from 5.45 billion pairs to 12,861
   projected cells.


## Accuracy and scope

The accelerated signed-distance path is benchmarked on watertight meshes. It
uses the original C++ implementation as its reference, with the error limits
listed above. For non-watertight input, signed values are not reliable before
repair in either backend; use `fix=True` to retain the original repair pipeline.

Small grids (`size < 64`) use the C++ backend because CUDA launch overhead can
exceed the work saved. Keep `backend="cpu"` available for CPU-only hosts and
for a direct reference result.


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
