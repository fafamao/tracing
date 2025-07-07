#include "render.cuh"

__global__ void render_device(int max_x, int max_y, Camera **cam, hittable **world, char *const pixel_buffer)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    Color col(0, 0, 0);
    for (int s = 0; s < PIXEL_NEIGHBOR; s++)
    {
        Ray r = (*cam)->get_ray(i, j);
        col += (*cam)->ray_color_device(r, MAX_DEPTH, world);
    }
    col *= (*cam)->_pixel_scale;
    col.write_color(i, j, pixel_buffer);
}