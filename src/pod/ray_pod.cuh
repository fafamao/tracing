#ifndef RAY_POD_CUH_
#define RAY_POD_CUH_

#include "vec3_pod.cuh"
namespace cuda_device
{
    struct Ray
    {
        Vec3 origin;
        Vec3 direction;
        float time;
    };

    // Calculates a point along the ray at distance t.
    __device__ inline Vec3 ray_at(const Ray &r, float t)
    {
        return r.origin + t * r.direction;
    }
}
#endif // RAY_POD_CUH_