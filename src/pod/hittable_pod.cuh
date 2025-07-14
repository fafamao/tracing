#ifndef HITTABLE_POD_CUH_
#define HITTABLE_POD_CUH_

#include "hittable_list_pod.cuh"
#include "interval_pod.cuh"
#include "material_pod.cuh"
#include "ray_pod.cuh"
#include "sphere_pod.cuh"

// Forward declaration to break circular dependency
struct BVHNode;
struct Sphere;

// The POD struct now contains the material data directly.
struct HitRecord {
  Vec3 p;
  Vec3 normal;
  Material mat;
  float t;
  bool front_face;
};

// Nodes for index and objects for real data
// For memory efficiency and cache performance
struct BVH {
  BVHNode *nodes;    // Pointer to the flat array of BVH nodes of index
  Hittable *objects; // Pointer to the flat array of all scene objects
};

// An enum to identify every possible hittable object
enum ObjectType {
  SPHERE,
  HITTABLE_LIST,
  BVH_NODE
  // TODO: add more objects here
};

struct Hittable {
  ObjectType type;
  union {
    Sphere *sphere;
    HittableList *list;
    BVH *bvh;
  };
};

__device__ inline bool hittable_hit(const Hittable &object, const Ray &r,
                                    Interval &ray_t, HitRecord &rec);

// --- The 'bounding_box' Dispatcher Function ---
__device__ inline aabb hittable_bounding_box(const Hittable &object);

__device__ inline Hittable create_hittable_from_sphere(const Sphere &s);

// Sets the hit record's normal vector.
// The HitRecord is passed by reference to be modified.
__device__ inline void set_face_normal(HitRecord &rec, const Ray &r,
                                       const Vec3 &outward_normal);

#endif // HITTABLE_POD_CUH_