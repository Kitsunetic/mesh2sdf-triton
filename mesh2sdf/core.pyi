import numpy as np
from numpy.typing import ArrayLike, NDArray

def compute(
    vertices: ArrayLike, faces: ArrayLike, size: int = 128
) -> NDArray[np.float32]: ...
