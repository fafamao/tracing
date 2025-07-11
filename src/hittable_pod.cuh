#ifndef HITTABLE_POD_CUH_
#define HITTABLE_POD_CUH_

#include "ray_pod.cuh"
#include "vec3_pod.cuh"
#include "sphere_pod.cuh"
#include "interval_pod.cuh"
#include "material_pod.cuh"
#include "hittable_list_pod.cuh"
#include "bvh_node_pod.cuh"

// The POD struct now contains the material data directly.
struct HitRecord
{
    Vec3 p;
    Vec3 normal;
    Material mat;
    float t;
    bool front_face;
};

// Nodes for index and objects for real data
// For memory efficiency and cache performance
struct BVH
{
    BVHNode *nodes;    // Pointer to the flat array of BVH nodes of index
    Hittable *objects; // Pointer to the flat array of all scene objects
};

// Sets the hit record's normal vector.
// The HitRecord is passed by reference to be modified.
__device__ inline void set_face_normal(
    HitRecord &rec, const Ray &r, const Vec3 &outward_normal)
{
    // NOTE: the parameter `outward_normal` is assumed to have unit length.
    rec.front_face = dot(r.direction, outward_normal) < 0;
    rec.normal = rec.front_face ? outward_normal : -outward_normal;
}

// An enum to identify every possible hittable object
enum ObjectType
{
    SPHERE,
    HITTABLE_LIST,
    BVH_NODE
    // TODO: add more objects here
};

struct Hittable
{
    ObjectType type;
    union
    {
        Sphere sphere;
        HittableList *list; // For collections, pointers are often still necessary
        BVH *bvh;
    };
};

__device__ inline bool hittable_hit(
    const Hittable &object, const Ray &r, Interval &ray_t, HitRecord &rec)
{
    switch (object.type)
    {
    case SPHERE:
        return hit_sphere(object.sphere, r, ray_t, rec);
    case HITTABLE_LIST:
        return hit_hittable_list(object.list, r, ray_t, rec);
        break;
    case BVH_NODE:
        return hit_bvh(object.bvh->nodes, object.bvh->objects, 0, r, ray_t, rec);
        break;
    }
    return false;
}

// --- The 'bounding_box' Dispatcher Function ---
__device__ inline aabb hittable_bounding_box(const Hittable &object)
{
    switch (object.type)
    {
    case SPHERE:
        return bounding_box_sphere(object.sphere);
    case HITTABLE_LIST:
        // return bounding_box_hittable_list(object.list);
        break; // Placeholder
    case BVH_NODE:
        // return bounding_box_bvh_node(object.bvh);
        break; // Placeholder
    }
    // Return an empty box for unknown types
    return EMPTY_AABB;
}

#endif // HITTABLE_POD_CUH_