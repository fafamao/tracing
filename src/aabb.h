#ifndef AABB_H
#define AABB_H

#include "interval.h"
#include "ray.h"
#include "vec3.h"
#include <cmath>

class aabb
{
public:
    Interval x, y, z;

    __host__ __device__ aabb() {} // The default AABB is empty, since Intervals are empty by default.

    __host__ __device__ aabb(const Interval &x, const Interval &y, const Interval &z)
        : x(x), y(y), z(z) {}

    __host__ __device__ aabb(const Vec3 &a, const Vec3 &b)
    {
        // Treat the two points a and b as extrema for the bounding box, so we don't require a
        // particular minimum/maximum coordinate order.

        x = (a.get_x() <= b.get_x()) ? Interval(a.get_x(), b.get_x()) : Interval(b.get_x(), a.get_x());
        y = (a.get_y() <= b.get_y()) ? Interval(a.get_y(), b.get_y()) : Interval(b.get_y(), a.get_y());
        z = (a.get_z() <= b.get_z()) ? Interval(a.get_z(), b.get_z()) : Interval(b.get_z(), a.get_z());
    }

    __host__ __device__ aabb(const aabb &box1, const aabb &box2)
    {
        x = Interval(box1.x, box2.x);
        y = Interval(box1.y, box2.y);
        z = Interval(box1.z, box2.z);
    }

    __host__ __device__ const Interval &axis_interval(int n) const
    {
        if (n == 1)
            return y;
        if (n == 2)
            return z;
        return x;
    }

    __host__ __device__ bool hit(const Ray &r, Interval ray_t) const
    {
        const Vec3 &ray_orig = r.get_origin();
        const Vec3 &ray_dir = r.get_direction();

        for (int axis = 0; axis < 3; axis++)
        {
            const Interval &ax = axis_interval(axis);
            const double adinv = 1.0 / ray_dir[axis];

            auto t0 = (ax.min - ray_orig[axis]) * adinv;
            auto t1 = (ax.max - ray_orig[axis]) * adinv;

            double slab_min = fmin(t0, t1); // Find the smaller of t0, t1
            double slab_max = fmax(t0, t1); // Find the larger of t0, t1
            ray_t.min = fmax(ray_t.min, slab_min);
            ray_t.max = fmin(ray_t.max, slab_max);

            if (ray_t.max <= ray_t.min)
                return false;
        }
        return true;
    }
};

#endif // AABB_H