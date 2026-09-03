from ._autograd import compute_triton as compute_triton
from ._pytorch import compute_pytorch as compute_pytorch
from .compute import compute as compute

__version__ = "2.0.0"

__all__ = ["__version__", "compute", "compute_pytorch", "compute_triton"]
