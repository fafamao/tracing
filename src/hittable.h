#ifndef HITTABLE_H
#define HITTABLE_H

#include "vec3.h"
#include "ray.h"

using namespace vec3;
using namespace ray;

class hit_record
{
public:
    Vec3 p;
    Vec3 normal;
    double t;
};

class hittable
{
public:
    virtual ~hittable() = default;

    virtual bool hit(const Ray &r, double ray_tmin, double ray_tmax, hit_record &rec) const = 0;
};

#endif // HITTABLE_H