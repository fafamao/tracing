#include <cstdio>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void rand_init(curandState *state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curand_init(0, 0, 0, state); // Explicit seed
    }
}

int main()
{
    curandState *d_state;
    cudaMalloc((void **)&d_state, sizeof(curandState));
    rand_init<<<1, 1>>>(d_state);
    cudaDeviceSynchronize();
    cudaFree(d_state);
    return 0;
}