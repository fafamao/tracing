#ifndef MATERIAL_H_
#define MATERIAL_H_

#include "hittable.h"
#include "ray.h"
#include "color.h"
#include <curand_kernel.h>

typedef struct
{
    Vec3 p;
    Vec3 normal;
    bool front_face;
} record_content;

class Material
{
public:
    virtual ~Material() = default;

    __host__ virtual bool scatter(
        const Ray &r_in, const record_content &record, Color &attenuation, Ray &scattered) const = 0;
    __device__ virtual bool scatter(
        const Ray &r_in, const record_content &record, Color &attenuation, Ray &scattered, curandState *local_rand_state) const = 0;
};

class Lambertian : public Material
{
public:
    __host__ __device__ Lambertian(const Color &albedo) : albedo(albedo) {}

    __host__ bool scatter(const Ray &r_in, const record_content &record, Color &attenuation, Ray &scattered)
        const override
    {
        auto scatter_direction = record.normal + random_unit_vec_rejection_method();
        if (scatter_direction.near_zero())
            scatter_direction = record.normal;
        scattered = Ray(record.p, scatter_direction, r_in.get_time());
        attenuation = albedo;
        return true;
    }

    __device__ bool scatter(const Ray &r_in, const record_content &record, Color &attenuation, Ray &scattered, curandState *local_rand_state)
        const override
    {
        // TODO
        return true;
    }

private:
    Color albedo;
};

class Metal : public Material
{
public:
    __host__ __device__ Metal(const Color &albedo, double fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

    __host__ bool scatter(const Ray &r_in, const record_content &record, Color &attenuation, Ray &scattered)
        const override
    {
        Vec3 reflected = reflect(r_in.get_direction(), record.normal);
        // Fuzz reflection
        reflected = unit_vector(reflected) + (fuzz * random_unit_vec_rejection_method());
        scattered = Ray(record.p, reflected, r_in.get_time());
        attenuation = albedo;
        return (dot(scattered.get_direction(), record.normal) > 0);
    }

    __device__ bool scatter(const Ray &r_in, const record_content &record, Color &attenuation, Ray &scattered, curandState *local_rand_state)
        const override
    {
        return true;
    }

private:
    Color albedo;
    double fuzz;
};

class Dielectric : public Material
{
public:
    __host__ __device__ Dielectric(double refraction_index) : refraction_index(refraction_index) {}

    __host__ bool scatter(const Ray &r_in, const record_content &record, Color &attenuation, Ray &scattered)
        const override
    {
        attenuation = Color(1.0, 1.0, 1.0);
        double ri = record.front_face ? (1.0 / refraction_index) : refraction_index;

        Vec3 unit_direction = unit_vector(r_in.get_direction());
        double cos_theta = std::fmin(dot(-unit_direction, record.normal), 1.0);
        double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0;
        Vec3 direction;

        if (cannot_refract || reflectance(cos_theta, ri) > random_double())
            direction = reflect(unit_direction, record.normal);
        else
            direction = refract(unit_direction, record.normal, ri);

        scattered = Ray(record.p, direction, r_in.get_time());
        return true;
    }

    __device__ bool scatter(const Ray &r_in, const record_content &record, Color &attenuation, Ray &scattered, curandState *local_rand_state)
        const override
    {
        return true;
    }

private:
    // Refractive index in vacuum or air, or the ratio of the material's refractive index over
    // the refractive index of the enclosing media
    double refraction_index;

    __host__ __device__ static double reflectance(double cosine, double refraction_index)
    {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0 * r0;
        return r0 + (1 - r0) * std::pow((1 - cosine), 5);
    }
};

#endif // MATERIAL_H_
