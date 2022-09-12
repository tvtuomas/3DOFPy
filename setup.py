from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("successiveOverrelaxation3D.pyx", build_dir="build"), include_dirs=[numpy.get_include()]
)