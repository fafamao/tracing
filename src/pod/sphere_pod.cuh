#ifndef SPHERE_POD_CUH_
#define SPHERE_POD_CUH_

#include "aabb_pod.cuh"
#include "material_pod.cuh"
#include "vec3_pod.cuh"

namespace cuda_device
{

  // Break circular dependency
  struct HitRecord;

  struct Sphere
  {
    bool is_moving;
    float radius;
    Vec3 center0;
    Vec3 center_vec;
    Material mat;
  };

  Sphere create_static_sphere(const Vec3 &center, float radius,
                              const Material &mat);

  // Creates a moving sphere
  Sphere create_moving_sphere(const Vec3 &center0, const Vec3 &center1,
                              float radius, const Material &mat);

  // Gets the center of the sphere at a specific time for motion blur
  __device__ __host__ Vec3 sphere_center(const Sphere &s, float time);

  // Calculates the bounding box for a sphere
  __device__ __host__ aabb bounding_box_sphere(const Sphere &s);

  // The main ray-sphere intersection logic
  __device__ bool hit_sphere(const Sphere &s, const Ray &r, Interval ray_t,
                             HitRecord &rec);
}
#endif // SPHERE_POD_CUH_