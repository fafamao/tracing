#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "aabb.h"

class sphere : public hittable
{
public:
    sphere(const Vec3 &center, double radius, shared_ptr<Material> mat) : center(center, Vec3(0, 0, 0)), radius(std::fmax(0, radius)), mat(mat)
    {
        auto rvec = Vec3(radius, radius, radius);
        box = aabb(center - rvec, center + rvec);
    }
    sphere(const Vec3 &center1, const Vec3 &center2, double radius, shared_ptr<Material> mat) : center(center1, center2 - center1), radius(std::fmax(0, radius)), mat(mat)
    {
        auto rvec = Vec3(radius, radius, radius);
        aabb box1(center.at(0) - rvec, center.at(0) + rvec);
        aabb box2(center.at(1) - rvec, center.at(1) + rvec);
        box = aabb(box1, box2);
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

    aabb bounding_box() const override
    {
        return box;
    }

private:
    Ray center;
    double radius;
    shared_ptr<Material> mat;
    aabb box;
};

#endif // SPHERE_H