#ifndef AABB_CUH_
#define AABB_CUH_

#include "vec3_pod.cuh"
#include "interval_pod.cuh"
#include "ray_pod.cuh"
namespace cuda_device
{

    // The POD struct contains only the data members.
    struct aabb
    {
        Interval x, y, z;
    };

    // Represents an empty bounding box.
    const aabb EMPTY_AABB = {EMPTY_INTERVAL, EMPTY_INTERVAL, EMPTY_INTERVAL};

    // Creates a bounding box from two points.
    __device__ inline aabb create_aabb_from_points(const Vec3 &a, const Vec3 &b)
    {
        // Treat the two points a and b as extrema for the bounding box.
        Interval ix = (a.x <= b.x) ? Interval{a.x, b.x} : Interval{b.x, a.x};
        Interval iy = (a.y <= b.y) ? Interval{a.y, b.y} : Interval{b.y, a.y};
        Interval iz = (a.z <= b.z) ? Interval{a.z, b.z} : Interval{b.z, a.z};
        return aabb{ix, iy, iz};
    }

    // Creates a new bounding box that encloses two other bounding boxes.
    __device__ inline aabb create_aabb_from_boxes(const aabb &box1, const aabb &box2)
    {
        Interval ix = create_combined_interval(box1.x, box2.x);
        Interval iy = create_combined_interval(box1.y, box2.y);
        Interval iz = create_combined_interval(box1.z, box2.z);
        return aabb{ix, iy, iz};
    }

    // Returns the interval for a given axis (0=x, 1=y, 2=z).
    __device__ inline const Interval &aabb_axis_interval(const aabb &box, int n)
    {
        if (n == 1)
            return box.y;
        if (n == 2)
            return box.z;
        return box.x;
    }

    // Performs the ray-AABB intersection test.
    __device__ inline bool aabb_hit(const aabb &box, const Ray &r, Interval ray_t)
    {
        for (int axis = 0; axis < 3; axis++)
        {
            const Interval &ax = aabb_axis_interval(box, axis);
            // Accessing Vec3 components directly as x, y, z or via an index operator if you prefer
            const float ray_orig_axis = (&r.origin.x)[axis];
            const float ray_dir_axis = (&r.direction.x)[axis];

            const float adinv = 1.0f / ray_dir_axis;

            auto t0 = (ax.min - ray_orig_axis) * adinv;
            auto t1 = (ax.max - ray_orig_axis) * adinv;

            if (adinv < 0.0f)
            {
                // Swap t0 and t1 if the ray direction is negative for this axis
                float temp = t0;
                t0 = t1;
                t1 = temp;
            }

            if (t0 > ray_t.min)
                ray_t.min = t0;
            if (t1 < ray_t.max)
                ray_t.max = t1;

            if (ray_t.max <= ray_t.min)
                return false;
        }
        return true;
    }
}
#endif // AABB_CUH_