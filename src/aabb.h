#ifndef AABB_H
#define AABB_H

#include "interval.h"
#include "ray.h"
#include "vec3.h"

class aabb
{
public:
    Interval x, y, z;

    aabb() {} // The default AABB is empty, since Intervals are empty by default.

    aabb(const Interval &x, const Interval &y, const Interval &z)
        : x(x), y(y), z(z) {}

    aabb(const Vec3 &a, const Vec3 &b)
    {
        // Treat the two points a and b as extrema for the bounding box, so we don't require a
        // particular minimum/maximum coordinate order.

        x = (a[0] <= b[0]) ? Interval(a[0], b[0]) : Interval(b[0], a[0]);
        y = (a[1] <= b[1]) ? Interval(a[1], b[1]) : Interval(b[1], a[1]);
        z = (a[2] <= b[2]) ? Interval(a[2], b[2]) : Interval(b[2], a[2]);
    }

    const Interval &axis_Interval(int n) const
    {
        if (n == 1)
            return y;
        if (n == 2)
            return z;
        return x;
    }

    bool hit(const Ray &r, Interval ray_t) const
    {
        const Vec3 &ray_orig = r.get_origin();
        const Vec3 &ray_dir = r.get_direction();

        for (int axis = 0; axis < 3; axis++)
        {
            const Interval &ax = axis_Interval(axis);
            const double adinv = 1.0 / ray_dir[axis];

            auto t0 = (ax.min - ray_orig[axis]) * adinv;
            auto t1 = (ax.max - ray_orig[axis]) * adinv;

            if (t0 < t1)
            {
                if (t0 > ray_t.min)
                    ray_t.min = t0;
                if (t1 < ray_t.max)
                    ray_t.max = t1;
            }
            else
            {
                if (t1 > ray_t.min)
                    ray_t.min = t1;
                if (t0 < ray_t.max)
                    ray_t.max = t0;
            }

            if (ray_t.max <= ray_t.min)
                return false;
        }
        return true;
    }
};

#endif // AABB_H