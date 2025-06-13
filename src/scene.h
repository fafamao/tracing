#ifndef SCENE_H_
#define SCENE_H_

#include "hittable_list.h"
#include <curand_kernel.h>

__global__ void generate_scene_device(hittable **d_list, hittable **d_world, curandState *rand_state);
void generate_scene_host(hittable_list &world);

#endif // SCENE_H_
