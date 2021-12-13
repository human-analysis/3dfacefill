from setuptools import setup
import os
import torch
import sysconfig
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

_DEBUG = False
_DEBUG_LEVEL = 0
# extra_compile_args = []
# extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
# extra_compile_args.remove('-DNDEBUG')
# extra_compile_args.remove('-O3')
if(_DEBUG):
    extra_compile_args = ['-g', '-O0', '-lineinfo', '-DDEBUG=%s' % _DEBUG_LEVEL, '-UNDEBUG']
else:
    # extra_compile_args += ['-g', '-O3', '-DNDEBUG']
    extra_compile_args = {'cxx': ['-g'], 'nvcc': ['-O2']}
print(extra_compile_args)

setup(
    name="zbuffertri_batch",
    version="1.0.0",
    ext_modules=[
        CUDAExtension(
            "zbuffertri_batch",
            ["zbuffertri.cpp", "zbuffertri_implementation.cu"],
            extra_compile_args=extra_compile_args
            # include_dirs=include_dirs
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

# .with_options(no_python_abi_suffix=True)
