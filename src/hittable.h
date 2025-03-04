#ifndef HITTABLE_H
#define HITTABLE_H

#include <vector>
#include "ray.h"
#include "interval.h"
#include "material.h"
#include <memory>

using std::shared_ptr;

class hit_record
{
public:
    Vec3 p;
    Vec3 normal;
    shared_ptr<Material> mat;
    double t;
    bool front_face;

    void set_face_normal(const Ray &r, const Vec3 &outward_normal)
    {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.

        front_face = dot(r.get_direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable
{
public:
    virtual ~hittable() = default;

    virtual bool hit(const Ray &r, Interval &interval, hit_record &rec) const = 0;
};

#endif // HITTABLE_H