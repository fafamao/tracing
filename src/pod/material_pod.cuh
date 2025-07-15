#ifndef MATERIAL_POD_CUH_
#define MATERIAL_POD_CUH_

#include "color_pod.cuh"
#include "constants.h"
#include "ray_pod.cuh"
namespace cuda_device
{
  // Forward-declare the HitRecord to avoid circular dependencies if needed
  struct HitRecord;

  enum MaterialType
  {
    LAMBERTIAN,
    METAL,
    DIELECTRIC
  };

  struct Material
  {
    MaterialType type;
    Color albedo; // Common property

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

  // Helper function, formerly a private method in Dielectric
  __device__ inline float reflectance(float cosine, float refraction_index);

  __device__ bool scatter_lambertian(const Ray &r_in, const HitRecord &rec,
                                     Color &attenuation, Ray &scattered);

  __device__ bool scatter_metal(const Ray &r_in, const HitRecord &rec,
                                Color &attenuation, Ray &scattered);

  __device__ bool scatter_dielectric(const Ray &r_in, const HitRecord &rec,
                                     Color &attenuation, Ray &scattered);

  typedef bool (*ScatterFn)(const Ray &, const HitRecord &, Color &, Ray &);
  extern __device__ ScatterFn scatter_functions[];

  __device__ bool scatter(const Ray &r_in, const HitRecord &rec,
                          Color &attenuation, Ray &scattered);
}
#endif // MATERIAL_POD_CUH_