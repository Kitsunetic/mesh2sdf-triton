import numpy as np
import skimage.measure
import trimesh
from numpy.typing import NDArray


def _compute_grid(vertices: NDArray[np.generic], faces: NDArray[np.generic],
                  size: int, device: str
                  ) -> NDArray[np.float32]:
  from ._cuda import compute_cuda
  return compute_cuda(np.ascontiguousarray(vertices, dtype=np.float32),
                      np.ascontiguousarray(faces, dtype=np.uint32), size, device)


def compute(vertices: NDArray[np.generic], faces: NDArray[np.generic], size: int = 128,
            fix: bool = False, level: float = 0.015, return_mesh: bool = False,
            device: str = "cuda"
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
    device (str): CUDA device used by the PyTorch and Triton implementation.
  '''

  # compute sdf
  sdf = _compute_grid(vertices, faces, size, device)
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
  sdf = _compute_grid(mesh.vertices, mesh.faces, size, device)
  return (sdf, mesh) if return_mesh else sdf
