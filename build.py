import os
import sys
import shutil
from distutils.core import Distribution, Extension
from Cython.Build import build_ext, cythonize
import numpy

extension = Extension(
    '*',
    ['src/**/*.pyx'],
    extra_compile_args=['/d2FH4-'] if sys.platform == 'win32' else ['-O3'],
    include_dirs=[numpy.get_include()],
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
)

ext_modules = cythonize([extension])
dist = Distribution({'ext_modules': ext_modules})
cmd = build_ext(dist)
cmd.ensure_finalized()
cmd.run()

for output in cmd.get_outputs():
    relative_extension = os.path.join('src', os.path.relpath(output, cmd.build_lib))
    shutil.copyfile(output, relative_extension)
