#include "hittable_pod.cuh"
#include "bvh_node_pod.cuh"
#include "vec3_pod.cuh"
namespace cuda_device
{

    __device__ bool hittable_hit(const Hittable &object, const Ray &r,
                                 Interval &ray_t, HitRecord &rec)
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
    __device__ aabb hittable_bounding_box(const Hittable &object)
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

    __device__ Hittable create_hittable_from_sphere(Sphere *s)
    {
        Hittable h;
        h.type = SPHERE;
        h.sphere = s;
        return h;
    }

    __device__ void set_face_normal(HitRecord &rec, const Ray &r,
                                    const Vec3 &outward_normal)
    {
        // NOTE: the parameter `outward_normal` is assumed to have unit length.
        rec.front_face = dot(r.direction, outward_normal) < 0;
        rec.normal = rec.front_face ? outward_normal : -outward_normal;
    }
}