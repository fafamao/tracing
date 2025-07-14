#ifndef RENDER_CUH_
#define RENDER_CUH_

#include "hittable.h"
#include "camera.h"
#include "vec3.h"
#include "constants.h"
#include "random_number_generator.cuh"

__global__ void render_device(int max_x, int max_y, Camera **cam, hittable **world, char *pixel_buffer);

#endif // RENDER_CUH_