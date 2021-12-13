#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> zbuffertri_cuda_forward(
                                at::Tensor s2d,
                                at::Tensor tri,
                                at::Tensor visible,
                                int img_size = 224);

// C++ interface

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> zbuffertri_forward(at::Tensor s2d, at::Tensor tri, at::Tensor visible, int img_size = 224)
{
    CHECK_INPUT(s2d);
    CHECK_INPUT(tri);
    CHECK_INPUT(visible);

    return zbuffertri_cuda_forward(s2d, tri, visible, img_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &zbuffertri_forward, "ZBufferTri Operation (CUDA)",
        py::arg("vertex2d"), py::arg("tri"), py::arg("visibility"), py::arg("img_size") = 224);
}
