from torch.utils.cpp_extension import load

zbuffertri_batch = load(name='zbuffertri_batch', sources=["zbuffertri.cpp", "zbuffertri_implementation.cu"], extra_cflags=['-O2'], verbose=True)
