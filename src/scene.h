#ifndef SCENE_H_
#define SCENE_H_

#include "hittable_list.h"
#include <curand_kernel.h>
#include "scene.h"
#include "sphere.h"
#include "material.h"
#include "bvh_node.h"
#include "camera.h"
#include "random_number_generator.cuh"

__global__ void generate_scene_device(hittable **d_list, hittable **d_world, curandState *rand_state, Camera **camera);
__global__ void free_scene(hittable **d_list, hittable **d_world, Camera **d_camera);
void generate_scene_host(hittable_list &world);

#endif // SCENE_H_
