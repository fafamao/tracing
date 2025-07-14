#ifndef CAMERA_POD_CUH_
#define CAMERA_POD_CUH_

#include "vec3_pod.cuh"
#include "ray_pod.cuh"
#include "constants.h"
namespace cuda_device
{

    // The simple POD struct with only the data needed by the kernel.
    struct CameraData
    {
        Vec3 top_left_pixel;
        Vec3 unit_vec_u;
        Vec3 unit_vec_v;
        Vec3 camera_origin;
        float pixel_scale;
    };

    // Generates a random offset for anti-aliasing.
    __device__ inline Vec3 sample_square_device()
    {
        // Generate vectors pointing to ([-0.5,0.5], [-0.5, 0.5], 0)
        return Vec3{random_float() - 0.5f, random_float() - 0.5f, 0};
    }

    // The core ray generation function, now operating on the POD struct.
    __device__ inline Ray get_ray_device(const CameraData &cam, int i, int j)
    {
        Vec3 offset = sample_square_device();
        Vec3 pixel_sample = cam.top_left_pixel + ((float(i) + offset.x) * cam.unit_vec_u) + ((float(j) + offset.y) * cam.unit_vec_v);

        Vec3 ray_origin = cam.camera_origin;
        Vec3 ray_direction = pixel_sample - ray_origin;

        // A time of 0 for a static scene.
        return Ray{ray_origin, ray_direction, 0.0f};
    }
}
#endif // CAMERA_POD_CUH_