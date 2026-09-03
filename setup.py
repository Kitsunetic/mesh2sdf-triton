from setuptools import find_packages, setup

__version__ = "2.0.0"


with open("README.md", "r", encoding="utf-8") as fid:
  long_description = fid.read()

setup(
    name="mesh2sdf-triton",
    version=__version__,
    author="Peng-Shuai Wang",
    author_email="wangps@hotmail.com",
    url="https://github.com/Kitsunetic/mesh2sdf-triton",
    description="GPU signed-distance fields from watertight meshes with PyTorch and Triton",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["mesh2sdf_triton", "mesh2sdf_triton.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "trimesh",
        "scikit-image",
        "torch>=2.2",
        "triton>=2.2",
    ],
    license="MIT",
    license_files=["LICENSE"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
)
