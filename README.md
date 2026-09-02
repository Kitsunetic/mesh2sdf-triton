# Mesh2SDF

[![Downloads](https://static.pepy.tech/badge/mesh2sdf)](https://pepy.tech/project/mesh2sdf)
[![Downloads](https://static.pepy.tech/badge/mesh2sdf/month)](https://pepy.tech/project/mesh2sdf)
[![PyPI](https://img.shields.io/pypi/v/mesh2sdf)](https://pypi.org/project/mesh2sdf/)


Converts an input mesh to a signed distance field. It can work with arbitrary
meshes, even **non-watertight** meshes from ShapeNet.

`mesh2sdf` is used in our paper
[Dual Octree Graph Networks (SIGGRAPH 2022)](https://wang-ps.github.io/dualocnn)
to generate the training data.
Please cite our paper if you find the code useful for your research.


## Installation

`mesh2sdf` depends on [pybind11](https://github.com/pybind/pybind11), and C++
compilers are needed to build the code. Supported compilers are listed
[here](https://github.com/pybind/pybind11#supported-compilers).

- Install via the following command:
    ``` shell
    pip install mesh2sdf
    ```

- Alternatively, install from the source code via the following commands.
    ``` shell
    git clone https://github.com/Kitsunetic/mesh2sdf-triton.git
    pip install ./mesh2sdf
    ```

- To enable CUDA acceleration on Linux, install the optional PyTorch and Triton
  dependencies:
    ``` shell
    pip install "mesh2sdf[cuda]"
    ```

## Example

After installing `mesh2sdf`, run the following command to process an input mesh
from ShapeNet:

```shell
python example/test.py
```

![Example of a mesh from ShapeNet](https://raw.githubusercontent.com/wang-ps/mesh2sdf/master/example/data/result.png)


## CUDA/Triton acceleration

For grids of size 64 or larger, `backend="auto"` uses the CUDA path when
PyTorch, Triton, and a CUDA device are available. Use `backend="cpu"` to force
the original C++ implementation or `backend="cuda"` to require acceleration.

The CUDA path does not build a CUDA extension or require `nvcc`. Triton JIT
compiles kernels on their first use for a GPU and problem shape; subsequent
calls reuse the compiled kernels. This means first-call latency is higher than
the warm measurements below.

The implementation preserves the original SDFGen rules while moving the
face-dependent work to the GPU:

- It computes C++-compatible triangle bounds on the GPU in float64, preserving
  the original `trunc`, `ceil`, `floor`, and clamp rules.
- Each triangle initializes only its inclusive narrow-band AABB. A packed
  `(squared_distance, triangle_id)` atomic minimum preserves the original
  nearest-triangle and first-triangle tie behavior.
- The eight-direction fast sweep is the same dependency order as the C++
  implementation.
- Sign intersections traverse each triangle's projected YZ bounds rather than
  every triangle/grid-line pair, avoiding face-count-linear work and large
  flattened launch indices.


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


## How does it work?

For watertight meshes, `mesh2sdf.compute` automatically uses CUDA when PyTorch,
Triton, and a CUDA device are available. Grids smaller than 64 use the C++ path
because GPU launch overhead is larger than the work saved.

```python
sdf = mesh2sdf.compute(vertices, faces, size=128, backend="cuda", device="cuda:0")
```

- Given an input mesh, we first compute the **unsigned** distance field with the
  fast sweeping algorithm implemented by
  [Christopher Batty (SDFGen)](https://github.com/christopherbatty/SDFGen).
  Note that the unsigned distance field can always be reliably and accurately
  computed even though the input mesh is non-watertight.

- Then we extract the level sets with a small value **d** with the marching cube
  algorithm. The extracted level sets are represented with triangle meshes and
  are guaranteed to be manifold.

- There exist multiple connected components in the extracted meshes, and we only
  keep the mesh with the largest bounding box.

- Compute the signed distance field again with the kept triangle mesh as the
  final output. In this way, the signed distance field (SDF) is computed for a
  non-watertight input mesh.


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
