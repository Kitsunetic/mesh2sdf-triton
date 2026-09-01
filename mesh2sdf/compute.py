from importlib.util import find_spec
from typing import Literal

import numpy as np
import skimage.measure
import trimesh
from numpy.typing import NDArray
from typing_extensions import assert_never

from .core import compute as compute_core

Backend = Literal["auto", "cpu", "cuda"]


class InvalidBackendError(ValueError):
  """Raised when an unknown compute backend is requested."""


def _parse_backend(backend: Backend) -> Backend:
  match backend:
    case "auto" | "cpu" | "cuda":
      return backend
    case unreachable:
      assert_never(unreachable)


def _cuda_available() -> bool:
  if find_spec("torch") is None or find_spec("triton") is None:
    return False
  from ._cuda import is_available
  return is_available()


def _compute_grid(vertices: NDArray[np.generic], faces: NDArray[np.generic],
                  size: int, backend: Backend, device: str
                  ) -> NDArray[np.float32]:
  use_cuda = size >= 64 and (
      backend == "cuda" or (backend == "auto" and _cuda_available()))
  if not use_cuda:
    return compute_core(vertices, faces, size)
  from ._cuda import compute_cuda
  return compute_cuda(np.ascontiguousarray(vertices, dtype=np.float32),
                      np.ascontiguousarray(faces, dtype=np.uint32), size, device)


def compute(vertices: NDArray[np.generic], faces: NDArray[np.generic], size: int = 128,
            fix: bool = False, level: float = 0.015, return_mesh: bool = False,
            backend: Backend = "auto", device: str = "cuda"
            ) -> NDArray[np.float32] | tuple[NDArray[np.float32], trimesh.Trimesh]:
  r''' Converts a input mesh to signed distance field (SDF).

  Args:
    vertices (np.ndarray): The vertices of the input mesh, the vertices MUST be
        in range [-1, 1].
    faces (np.ndarray): The faces of the input mesh.
    size (int): The resolution of the resulting SDF.
    fix (bool): If the input mesh is not watertight, set :attr:`fix` as True.
    level (float): The value used to extract level sets when :attr:`fix` is True,
        with a default value of 0.015 (as a reference 2/128 = 0.015625). And the
        recommended default value is 2/size.
    return_mesh (bool): If True, also return the fixed mesh.
    backend (str): ``"auto"`` uses CUDA when PyTorch and Triton are available,
        ``"cpu"`` selects the original C++ implementation, and ``"cuda"``
        requires the accelerated implementation.
    device (str): PyTorch CUDA device used by the accelerated implementation.
  '''

  selected_backend = _parse_backend(backend)

  # compute sdf
  sdf = _compute_grid(vertices, faces, size, selected_backend, device)
  if not fix:
    return (sdf, trimesh.Trimesh(vertices, faces)) if return_mesh else sdf

  # NOTE: the negative value is not reliable if the mesh is not watertight
  sdf = np.abs(sdf)
  vertices, faces, _, _ = skimage.measure.marching_cubes(sdf, level)

  # keep the max component of the extracted mesh
  mesh = trimesh.Trimesh(vertices, faces)
  components = mesh.split(only_watertight=False)
  bbox = []
  for c in components:
    bbmin = c.vertices.min(0)
    bbmax = c.vertices.max(0)
    bbox.append((bbmax - bbmin).max())
  max_component = np.argmax(bbox)
  mesh = components[max_component]
  mesh.vertices = mesh.vertices * (2.0 / size) - 1.0  # normalize it to [-1, 1]

  # re-compute sdf
  sdf = _compute_grid(mesh.vertices, mesh.faces, size, selected_backend, device)
  return (sdf, mesh) if return_mesh else sdf
