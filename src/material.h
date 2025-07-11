#ifndef MATERIAL_H_
#define MATERIAL_H_

#include "hittable.h"
#include "ray.h"
#include "color.h"
#include <curand_kernel.h>
#include <cuda_runtime.h>

typedef struct
{
    Vec3 p;
    Vec3 normal;
    bool front_face;
} record_content;

class Material
{
public:
    // virtual ~Material() = default;

    __host__ __device__ virtual bool scatter(
        const Ray &r_in, const record_content &record, Color &attenuation, Ray &scattered) const = 0;
};

class Lambertian : public Material
{
public:
    __host__ __device__ Lambertian(const Color &albedo) : albedo(albedo) {}

    __host__ __device__ bool scatter(const Ray &r_in, const record_content &record, Color &attenuation, Ray &scattered)
        const override
    {
        auto scatter_direction = record.normal + random_unit_vec_rejection_method();
        if (scatter_direction.near_zero())
            scatter_direction = record.normal;
        scattered = Ray(record.p, scatter_direction, r_in.get_time());
        attenuation = albedo;
        return true;
    }

private:
    Color albedo;
};

class Metal : public Material
{
public:
    __host__ __device__ Metal(const Color &albedo, float fuzz) : albedo(albedo), fuzz(fuzz < 1.0f ? fuzz : 1.0f) {}

    __host__ __device__ bool scatter(const Ray &r_in, const record_content &record, Color &attenuation, Ray &scattered)
        const override
    {
        Vec3 reflected = reflect(r_in.get_direction(), record.normal);
        // Fuzz reflection
        reflected = unit_vector(reflected) + (fuzz * random_unit_vec_rejection_method());
        scattered = Ray(record.p, reflected, r_in.get_time());
        attenuation = albedo;
        return (dot(scattered.get_direction(), record.normal) > 0);
    }

private:
    Color albedo;
    float fuzz;
};

class Dielectric : public Material
{
public:
    __host__ __device__ Dielectric(float refraction_index) : refraction_index(refraction_index) {}

    __host__ __device__ bool scatter(const Ray &r_in, const record_content &record, Color &attenuation, Ray &scattered)
        const override
    {
        attenuation = Color(1.0f, 1.0f, 1.0f);
        float ri = record.front_face ? (1.0f / refraction_index) : refraction_index;

        Vec3 unit_direction = unit_vector(r_in.get_direction());
        float cos_theta = fmin(dot(-unit_direction, record.normal), 1.0f);
        float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0f;
        Vec3 direction;

        if (cannot_refract || reflectance(cos_theta, ri) > random_float())
            direction = reflect(unit_direction, record.normal);
        else
            direction = refract(unit_direction, record.normal, ri);

        scattered = Ray(record.p, direction, r_in.get_time());
        return true;
    }

private:
    // Refractive index in vacuum or air, or the ratio of the material's refractive index over
    // the refractive index of the enclosing media
    float refraction_index;

    __host__ __device__ static float reflectance(float cosine, float refraction_index)
    {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0 * r0;

        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
};

#endif // MATERIAL_H_
