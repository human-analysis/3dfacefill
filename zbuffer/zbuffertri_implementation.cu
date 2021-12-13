#include <ATen/ATen.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#ifndef ERROR_CHECKING
#define ERROR_CHECKING 1
#endif /* ifndef ERROR_CHECKING */

// Macro for checking cuda errors following a cuda launch or api call
#if ERROR_CHECKING == 1
#define cudaCheckError()                                                       \
  {                                                                            \
    cudaDeviceSynchronize();                                                   \
    cudaError_t e = cudaGetLastError();                                        \
    if (e != cudaSuccess) {                                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,                 \
             cudaGetErrorString(e));                                           \
      exit(0);                                                                 \
    }                                                                          \
  }
#else
#define cudaCheckError()
#endif

#define max(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a > _b ? _a : _b; })

#define min(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a < _b ? _a : _b; })

#define min3(a,b,c) (min(min(a,b), c))

#define max3(a,b,c) (max(max(a,b), c))

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


template <typename scalar_t> __global__ void zbuffertri_cuda_forward_kernel(const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> s2d,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> tri,
        const torch::PackedTensorAccessor<bool,2,torch::RestrictPtrTraits,size_t> visible,
        torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> out,
        torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> zbuffer,
        torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> barycoords,
        const int img_size, const size_t tri_num, const size_t vertex_num
)
{
    // batch index
    const int n = blockIdx.y;
    // column index
    const int c = blockIdx.x * blockDim.x + threadIdx.x; // was c earlier
    float eps = 1e-8;

    for(int i=c; i<tri_num; i+=blockDim.x*gridDim.x) //check for i<tri_num
    {
        if(visible[n][i])
        {
            int vt1 = tri[0][i];
            int vt2 = tri[1][i];
            int vt3 = tri[2][i];

            float u1 = s2d[n][0][vt1];
            float v1 = s2d[n][1][vt1];
            float u2 = s2d[n][0][vt2];
            float v2 = s2d[n][1][vt2];
            float u3 = s2d[n][0][vt3];
            float v3 = s2d[n][1][vt3];

            float z1 = s2d[n][2][vt1];
            float z2 = s2d[n][2][vt2];
            float z3 = s2d[n][2][vt3];

            // assign umin=min(0, umin); umax=max(img_size, umax) and same for v to enable processing for partially visible triangles too
            int umin = int(ceil(double(min3(u1, u2, u3))));
            int umax = int(floor(double(max3(u1, u2, u3))));
            int vmin = int(ceil(double(min3(v1, v2, v3))));
            int vmax = int(floor(double(max3(v1, v2, v3))));

            float r = (s2d[n][2][vt1] + s2d[n][2][vt2] + s2d[n][2][vt3])/3;
            float min_r = min3(z1, z2, z3);

            for(int u=max(umin,0); u<=min(umax,img_size-1); u++)
            {
                for(int v=max(vmin,0); v<=min(vmax,img_size-1); v++)
                {
                    if(min_r > zbuffer[n][u][v])
                    {
                        float u_13 = u1 - u3;
                        float u_23 = u2 - u3;
                        float v_13 = v1 - v3;
                        float v_23 = v2 - v3;

                        float inv_deno = 1 / (u_13 * v_23 - u_23 * v_13 + eps);

                        float u_u3 = u - u3;
                        float v_v3 = v - v3;

                        float lambda1 = (v_23 * u_u3 - u_23 * v_v3) * inv_deno;
                        float lambda2 = (u_13 * v_v3 - v_13 * u_u3) * inv_deno;
                        float lambda3 = 1 - lambda1 - lambda2;

                        if(lambda1 > -eps && lambda1 < (1+eps) && lambda2 > -eps && lambda2 < (1+eps) && lambda3 > -eps && lambda3 < (1+eps))
                        {
                            zbuffer[n][u][v] = r;
                            out[n][u][v] = i;
                            barycoords[n][u][v][0] = lambda1;
                            barycoords[n][u][v][1] = lambda2;
                            barycoords[n][u][v][2] = lambda3;
                        }
                    }
                }
            }
        }
    }
}


template <typename scalar_t> __global__ void convert_to_mask(torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> zbuffer,
        torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> mask,
        const int img_size)
{
  const int n = blockIdx.z;
  const int r = blockIdx.y * blockDim.y + threadIdx.y; // earlier n
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x; // earlier c
  for(int i=r; i<img_size; i+=blockDim.y*gridDim.y)
  {
    for(int j=c; j<img_size; j+=blockDim.x*gridDim.x)
    {
    //if(i<img_size && j<img_size)
    //{
        if(zbuffer[n][i][j] == -INFINITY)
        {
            mask[n][i][j] = 0;
        }
        else
        {
            mask[n][i][j] = 1;
        }
    //}
    }
  }
}


/* Forward Function */
std::vector<at::Tensor> zbuffertri_cuda_forward(at::Tensor s2d, at::Tensor tri, at::Tensor visible, int img_size = 224)
{
    const int batch_size = s2d.size(0);
    const int tri_num = tri.size(1);
    const int vertex_num = s2d.size(1);

    auto out = torch::ones({batch_size, img_size, img_size}, torch::device(s2d.device())) * (tri_num-1);
    auto zbuffer = torch::ones({batch_size, img_size, img_size}, torch::device(s2d.device())) * (-INFINITY);
    auto barycoords = torch::zeros({batch_size, img_size, img_size, 3}, torch::device(s2d.device()));
    auto mask = torch::zeros({batch_size, img_size, img_size}, torch::device(s2d.device()));

    const int threads = 1024;
    //const int zbuffer_blocks = (tri_num + threads - 2) / threads; // try (tri_num + threads -1)
    const dim3 zbuffer_blocks((tri_num + threads - 1) / threads, batch_size);
    const dim3 mask_grid((img_size + 32 - 1) / 32, (img_size + 32 - 1) / 32, batch_size);
    const dim3 mask_threads((32, 32, 1));

    AT_DISPATCH_FLOATING_TYPES(zbuffer.type(), "zbuffer_tri_dispatch", ([&] {
        zbuffertri_cuda_forward_kernel<scalar_t><<<zbuffer_blocks, threads>>>(
                s2d.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                tri.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                visible.packed_accessor<bool,2,torch::RestrictPtrTraits,size_t>(),
                out.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                zbuffer.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                barycoords.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
                img_size, tri_num, vertex_num);
    }));

    AT_DISPATCH_FLOATING_TYPES(zbuffer.type(), "convert_to_mask", ([&] {
        convert_to_mask<scalar_t><<<mask_grid, mask_threads>>>(zbuffer.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                                               mask.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), img_size);
    }));

    /*gpuErrchk(cudaPeekAtLastError());
    gpuErrchk( cudaDeviceSynchronize() );*/

    return {out, zbuffer, barycoords, mask};
}
