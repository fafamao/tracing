#ifndef SCENE_H_
#define SCENE_H_

#include "hittable_list.h"

__global__ hittable_list* generate_scene_device();
void generate_scene_host(hittable_list &world);


#endif // SCENE_H_
