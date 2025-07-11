#ifndef CAMERA_H_
#define CAMERA_H_

#include "vec3.h"
#include "constants.h"
#include "color.h"
#include "hittable_list.h"
#include "material.h"
#include "../utility/thread_pool.h"
#include <functional>
#include <chrono>
#include <thread>

class Camera
{
private:
    Vec3 _top_left_pixel, _unit_vec_v, _unit_vec_u, _camera;
    Vec3 u, v, w; // Camera frame basis vectors
    __host__ __device__ void initialize();
    ThreadPool *thread_pool;

public:
    float _pixel_scale;
    // Angle between z direction ray and ray between origin and top edge of viewport * 2
    float vfov = 90;
    Vec3 lookfrom = Vec3(0, 0, 0); // Point camera is looking from
    Vec3 lookat = Vec3(0, 0, -1);  // Point camera is looking at
    Vec3 vup = Vec3(0, 1, 0);      // Camera-relative "up" direction

    Camera(Vec3 &origin, Vec3 &dest, Vec3 &up, ThreadPool *tp) : lookfrom(origin), lookat(dest), vup(up), thread_pool(tp)
    {
        initialize();
    }
    __device__ Camera(Vec3 &origin, Vec3 &dest, Vec3 &up) : lookfrom(origin), lookat(dest), vup(up)
    {
        initialize();
    }
    __host__ __device__ ~Camera()
    {
        printf("Camera: destructor called\n");
    }
    void render(const hittable_list &world, char *ptr);

    Color ray_color(const Ray &r, const int depth, const hittable_list &world);
    __device__ Color ray_color_device(const Ray &r, const int depth, hittable **world);

    // Generate vectors pointing to ([-0.5,0.5], [-0.5, 0.5], 0)
    __host__ __device__ Vec3 sample_square() const
    {
        // TODO: cuda random number generation
        return Vec3(random_float() - 0.5, random_float() - 0.5, 0);
    };

    __host__ __device__ Ray get_ray(int i, int j) const
    {
        auto offset = sample_square();
        auto pixel_sample = _top_left_pixel + ((i + offset.get_x()) * _unit_vec_u) + ((j + offset.get_y()) * _unit_vec_v);

        auto ray_origin = _camera;
        auto ray_direction = pixel_sample - ray_origin;

        auto random_time = 0; // random_float();

        return Ray(ray_origin, ray_direction, random_time);
    };
};

#endif // CAMERA_H_