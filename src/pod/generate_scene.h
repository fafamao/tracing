#ifndef POD_GENERATE_SCENE_H_
#define POD_GENERATE_SCENE_H_

#include "color_pod.cuh"
#include "hittable_pod.cuh"
#include "material_pod.cuh"
#include "vec3_pod.cuh"
#include <vector>

// Generate world from objects into hittable
std::vector<Hittable> generate_world();

// Generate lambertian material
Material create_lambertian_material(const Color &color);

// Generate metal material
Material create_metal_material(const Color &color, float fuzz);

// Generate glass material
Material create_dielectric_material(float index_of_refraction);

#endif // POD_GENERATE_SCENE_H_