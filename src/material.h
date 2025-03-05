#ifndef MATERIAL_H_
#define MATERIAL_H_

#include "hittable.h"
#include "ray.h"
#include "color.h"

class Material
{
public:
    virtual ~Material() = default;

    virtual bool scatter(
        const Ray &r_in, const Vec3 &normal, const Vec3 &p, Color &attenuation, Ray &scattered) const
    {
        return false;
    }
};

class Lambertian : public Material
{
public:
    Lambertian(const Color &albedo) : albedo(albedo) {}

    bool scatter(const Ray &r_in, const Vec3 &normal, const Vec3 &p, Color &attenuation, Ray &scattered)
        const override
    {
        auto scatter_direction = normal + random_unit_vec_rejection_method();
        if (scatter_direction.near_zero())
            scatter_direction = normal;
        scattered = Ray(p, scatter_direction);
        attenuation = albedo;
        return true;
    }

private:
    Color albedo;
};

class Metal : public Material
{
public:
    Metal(const Color &albedo, double fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

    bool scatter(const Ray &r_in, const Vec3 &normal, const Vec3 &p, Color &attenuation, Ray &scattered)
        const override
    {
        Vec3 reflected = reflect(r_in.get_direction(), normal);
        // Fuzz reflection
        reflected = unit_vector(reflected) + (fuzz * random_unit_vec_rejection_method());
        scattered = Ray(p, reflected);
        attenuation = albedo;
        return (dot(scattered.get_direction(), normal) > 0);
    }

private:
    Color albedo;
    double fuzz;
};

#endif // MATERIAL_H_
