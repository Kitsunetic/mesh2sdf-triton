import os
import sys
import time
from typing import cast

import numpy as np
import trimesh

import mesh2sdf_triton

filename = sys.argv[1] if len(sys.argv) > 1 else  \
    os.path.join(os.path.dirname(__file__), 'data', 'plane.obj')

mesh_scale = 0.8
size = 128
level = 2 / size

mesh = cast(trimesh.Trimesh, trimesh.load(filename, force='mesh'))

# normalize mesh
vertices = mesh.vertices
bbmin = vertices.min(0)
bbmax = vertices.max(0)
center = (bbmin + bbmax) * 0.5
scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
vertices = (vertices - center) * scale

# fix mesh
t0 = time.time()
sdf, mesh = mesh2sdf_triton.compute(
    vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)
t1 = time.time()

# output
mesh.vertices = mesh.vertices / scale + center
mesh.export(filename[:-4] + '.fixed.obj')
np.save(filename[:-4] + '.npy', sdf)
print(f"It takes {t1 - t0:.4f} seconds to process {filename}")
