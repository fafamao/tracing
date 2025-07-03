#include "render.cuh"

extern __device__ curandState *render_rand_state_global;

__global__ void render_device(int max_x, int max_y, Camera **cam, hittable **world)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = render_rand_state_global[pixel_index];
    Color col(0, 0, 0);
    for (int s = 0; s < PIXEL_NEIGHBOR; s++)
    {
        Ray r = (*cam)->get_ray(i, j);
        col += (*cam)->ray_color_device(r, MAX_DEPTH, world);
    }
    col *= (*cam)->_pixel_scale;
}