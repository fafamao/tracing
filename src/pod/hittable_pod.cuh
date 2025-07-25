#ifndef HITTABLE_POD_CUH_
#define HITTABLE_POD_CUH_

#include "hittable_list_pod.cuh"
#include "interval_pod.cuh"
#include "material_pod.cuh"
#include "ray_pod.cuh"
#include "sphere_pod.cuh"
namespace cuda_device
{

  // Forward declaration to break circular dependency
  struct Sphere;

  // The POD struct now contains the material data directly.
  struct HitRecord
  {
    Vec3 p;
    Vec3 normal;
    Material mat;
    float t;
    bool front_face;
  };

  // An enum to identify every possible hittable object
  enum ObjectType
  {
    SPHERE
  };

  struct Hittable
  {
    ObjectType type;
    Sphere sphere;
  };

  __device__ bool hittable_hit(const Hittable &object, const Ray &r,
                               Interval &ray_t, HitRecord &rec);

  // --- The 'bounding_box' Dispatcher Function ---
  __device__ __host__ aabb hittable_bounding_box(const Hittable &object);

  __device__ __host__ Hittable create_hittable_from_sphere(Sphere &s);

  // Sets the hit record's normal vector.
  // The HitRecord is passed by reference to be modified.
  __device__ __host__ void set_face_normal(HitRecord &rec, const Ray &r,
                                           const Vec3 &outward_normal);

  inline const char *object_type_to_string(ObjectType type)
  {
    switch (type)
    {
    case SPHERE:
      return "Sphere";
    // Add other types here as you create them
    default:
      return "Unknown";
    }
  }
}
#endif // HITTABLE_POD_CUH_