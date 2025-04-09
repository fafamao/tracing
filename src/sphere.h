#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"

class sphere : public hittable
{
public:
    sphere(const Vec3 &center, double radius, shared_ptr<Material> mat) : center(center, Vec3(0, 0, 0)), radius(std::fmax(0, radius)), mat(mat)
    {
        // TODO: dependency injection instead
    }
    sphere(const Vec3 &center1, const Vec3 &center2, double radius, shared_ptr<Material> mat) : center(center1, center2 - center1), radius(std::fmax(0, radius)), mat(mat)
    {
        // TODO: dependency injection instead
    }

    bool hit(const Ray &r, Interval &interval, hit_record &rec) const override
    {
        Vec3 current_center = center.at(r.get_time());
        Vec3 oc = current_center - r.get_origin();
        auto a = r.get_direction().length_squared();
        auto h = dot(r.get_direction(), oc);
        auto c = oc.length_squared() - radius * radius;

        auto discriminant = h * h - a * c;
        if (discriminant < 0)
            return false;

        auto sqrtd = std::sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        // TODO: optimization to let root fall within t range
        auto root = (h - sqrtd) / a;
        if (!interval.surrounds(root))
        {
            root = (h + sqrtd) / a;
            if (!interval.surrounds(root))
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        Vec3 outward_normal = (rec.p - current_center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat = mat;

        return true;
    }

private:
    Ray center;
    double radius;
    shared_ptr<Material> mat;
};

#endif // SPHERE_H