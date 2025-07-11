#ifndef MATERIAL_POD_CUH_
#define MATERIAL_POD_CUH_

#include "color_pod.cuh"

#include "ray_pod.cuh"
#include "hittable_pod.cuh"
#include "constants.h"

// Forward-declare the HitRecord to avoid circular dependencies if needed
// struct HitRecord;

// 1. An enum to identify each material type
enum MaterialType
{
    LAMBERTIAN,
    METAL,
    DIELECTRIC
};

// 2. The unified POD struct for all material data
struct Material
{
    MaterialType type;
    Color albedo; // Common property

    // Union for type-specific data to save memory
    union
    {
        struct
        {
            float fuzz; // For Metal
        } metal;

        struct
        {
            float ir; // For Dielectric (Index of Refraction)
        } dielectric;
    };
};

// --- Scatter Functions (replaces virtual methods) ---

// Helper function, formerly a private method in Dielectric
__device__ inline float reflectance(float cosine, float refraction_index)
{
    // Use Schlick's approximation for reflectance
    auto r0 = (1 - refraction_index) / (1 + refraction_index);
    r0 = r0 * r0;
    return r0 + (1 - r0) * powf((1 - cosine), 5);
}

__device__ bool scatter_lambertian(
    const Ray &r_in, const HitRecord &rec, Color &attenuation, Ray &scattered)
{
    auto scatter_direction = rec.normal + random_unit_vector();

    // Catch degenerate scatter direction
    if (near_zero(scatter_direction))
    {
        scatter_direction = rec.normal;
    }

    scattered = Ray{rec.p, scatter_direction, r_in.time};
    attenuation = rec.mat.albedo;
    return true;
}

__device__ bool scatter_metal(
    const Ray &r_in, const HitRecord &rec, Color &attenuation, Ray &scattered)
{
    Vec3 reflected = reflect(unit_vector(r_in.direction), rec.normal);
    scattered = Ray{rec.p, reflected + rec.mat.metal.fuzz * random_unit_vector(), r_in.time};
    attenuation = rec.mat.albedo;
    return (dot(scattered.direction, rec.normal) > 0);
}

__device__ bool scatter_dielectric(
    const Ray &r_in, const HitRecord &rec, Color &attenuation, Ray &scattered)
{
    attenuation = Color{1.0f, 1.0f, 1.0f};
    float ri = rec.front_face ? (1.0f / rec.mat.dielectric.ir) : rec.mat.dielectric.ir;

    Vec3 unit_direction = unit_vector(r_in.direction);
    float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);
    float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

    bool cannot_refract = ri * sin_theta > 1.0f;
    Vec3 direction;

    if (cannot_refract || reflectance(cos_theta, ri) > random_float())
    {
        direction = reflect(unit_direction, rec.normal);
    }
    else
    {
        direction = refract(unit_direction, rec.normal, ri);
    }

    scattered = Ray{rec.p, direction, r_in.time};
    return true;
}

// 1. Define the function pointer type
typedef bool (*ScatterFn)(const Ray &, const HitRecord &, Color &, Ray &);

// 2. Create the table of scatter functions on the device
__device__ ScatterFn scatter_functions[] = {
    scatter_lambertian,
    scatter_metal,
    scatter_dielectric};

__device__ bool scatter(const Ray &r_in, const HitRecord &rec, Color &attenuation, Ray &scattered)
{
    // Call the correct function based on the material's type ID
    return scatter_functions[rec.mat.type](r_in, rec, attenuation, scattered);
}

#endif // MATERIAL_POD_CUH_