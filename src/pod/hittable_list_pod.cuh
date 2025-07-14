#ifndef HITTABLE_LIST_POD_CUH_
#define HITTABLE_LIST_POD_CUH_

#include "aabb_pod.cuh"

struct Hittable;
struct HitRecord;

struct HittableList {
  Hittable *objects;
  int size;
  aabb bbox;
};

// The main intersection logic for the list
__device__ inline bool hit_hittable_list(const HittableList *list, const Ray &r,
                                         Interval ray_t, HitRecord &rec);

// The bounding box function simply returns the pre-computed box
__device__ inline aabb bounding_box_hittable_list(const HittableList *list);

#endif // HITTABLE_LIST_POD_CUH_