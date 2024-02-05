import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(
    ext_modules=cythonize("lifesim2/core/processing/intensity_response.pyx"),
    include_dirs=[numpy.get_include()]
)
