#include "material_pod.cuh"
#include "hittable_pod.cuh"
namespace cuda_device
{
    // Helper function, formerly a private method in Dielectric
    __device__ inline float reflectance(float cosine, float refraction_index)
    {
        // Use Schlick's approximation for reflectance
        auto r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0 * r0;
        return r0 + (1 - r0) * powf((1 - cosine), 5);
    }

    __device__ bool scatter_lambertian(const Ray &r_in, const HitRecord &rec,
                                       Color &attenuation, Ray &scattered)
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

    __device__ bool scatter_metal(const Ray &r_in, const HitRecord &rec,
                                  Color &attenuation, Ray &scattered)
    {
        Vec3 reflected = reflect(unit_vector(r_in.direction), rec.normal);
        scattered = Ray{rec.p, reflected + rec.mat.metal.fuzz * random_unit_vector(),
                        r_in.time};
        attenuation = rec.mat.albedo;
        return (dot(scattered.direction, rec.normal) > 0);
    }

    __device__ bool scatter_dielectric(const Ray &r_in, const HitRecord &rec,
                                       Color &attenuation, Ray &scattered)
    {
        attenuation = Color{1.0f, 1.0f, 1.0f};
        float ri =
            rec.front_face ? (1.0f / rec.mat.dielectric.ir) : rec.mat.dielectric.ir;

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

    __device__ bool scatter(const Ray &r_in, const HitRecord &rec,
                            Color &attenuation, Ray &scattered)
    {
        // Call the correct function based on the material's type ID
        return scatter_functions[rec.mat.type](r_in, rec, attenuation, scattered);
    }

    __device__ ScatterFn scatter_functions[] = {scatter_lambertian, scatter_metal,
                                                scatter_dielectric};
}