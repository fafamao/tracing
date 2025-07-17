#ifndef POD_GENERATE_SCENE_H_
#define POD_GENERATE_SCENE_H_

#include "color_pod.cuh"
#include "hittable_pod.cuh"
#include "material_pod.cuh"
#include "vec3_pod.cuh"
#include <vector>
namespace cuda_device {

// Generate world from objects into hittable
std::vector<Hittable> generate_world();
} // namespace cuda_device
#endif // POD_GENERATE_SCENE_H_