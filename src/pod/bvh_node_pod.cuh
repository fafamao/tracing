#ifndef BVH_NODE_POD_CUH_
#define BVH_NODE_POD_CUH_

#include "aabb_pod.cuh"
#include "hittable_pod.cuh"
#include "interval_pod.cuh"
#include "ray_pod.cuh"

struct Hittable;

struct BVHNode
{
    aabb bbox;
    // Index into the BVH node array
    int left_child_idx;
    // Index into the BVH node array
    int right_child_idx;
    // For leaf nodes
    bool is_leaf;
    // Index into the main hittable object array
    int object_idx;
};

__device__ inline bool hit_bvh(const BVHNode *bvh_nodes,
                               const Hittable *objects, int node_idx,
                               const Ray &r, Interval ray_t, HitRecord &rec);

#endif // BVH_NODE_POD_CUH_