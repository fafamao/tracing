#include <random_number_generator.cuh>

__global__ void rand_init(curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curand_init(0, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_index = j * max_x + i;
    // Same id and different seed boosts performance
    curand_init(pixel_index, 0, 0, &render_rand_state_global[pixel_index]);
}