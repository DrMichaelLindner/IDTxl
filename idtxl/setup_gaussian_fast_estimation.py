from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

package = Extension("idtxl.gaussian_fast_estimation",
        ["idtxl/gaussian_fast_estimation.pyx"], include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize([package]))