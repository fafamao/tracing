#ifndef CAMERA_POD_CUH_
#define CAMERA_POD_CUH_

#include "constants.h"
#include "ray_pod.cuh"
#include "vec3_pod.cuh"

namespace cuda_device
{

    struct CameraData
    {
        Vec3 top_left_pixel;
        Vec3 unit_vec_u;
        Vec3 unit_vec_v;
        Vec3 camera_origin;
    };

    // Generates a random offset for anti-aliasing.
    __device__ inline Vec3 sample_square_device()
    {
        return Vec3{random_float() - 0.5f, random_float() - 0.5f, 0};
    }

    // The core ray generation function, now operating on the POD struct.
    __device__ inline Ray get_ray_device(const CameraData &cam, int i, int j)
    {
        Vec3 offset = sample_square_device();
        Vec3 ray_direction = cam.top_left_pixel +
                             ((float(i) + offset.x) * cam.unit_vec_u) +
                             ((float(j) + offset.y) * cam.unit_vec_v) - cam.camera_origin;

        return Ray{cam.camera_origin, ray_direction, 0.0f};
    }

    __device__ __host__ inline CameraData
    construct_camera(const Vec3 &origin, const Vec3 &dest, const Vec3 &up)
    {
        Vec3 w = unit_vector(origin - dest);
        Vec3 u = unit_vector(cross(up, w));
        Vec3 v = cross(w, u);

        float focal_len = length(origin - dest);
        float theta = degrees_to_radians(VFOV);
        float h = tanf(theta / 2.0f);
        float viewport_height = 2 * h * focal_len;
        float viewport_width =
            viewport_height * (float(PIXEL_WIDTH) / float(PIXEL_HEIGHT));
        Vec3 vec_u = u * viewport_width;
        Vec3 vec_v = -v * viewport_height;
        Vec3 unit_vec_u = vec_u / PIXEL_WIDTH;
        Vec3 unit_vec_v = vec_v / PIXEL_HEIGHT;
        Vec3 top_left_pixel = origin - focal_len * w - vec_u / 2 - vec_v / 2 +
                              unit_vec_u / 2 + unit_vec_v / 2;

        CameraData camera = {top_left_pixel, unit_vec_u, unit_vec_v, origin};
        return camera;
    }
} // namespace cuda_device
#endif // CAMERA_POD_CUH_