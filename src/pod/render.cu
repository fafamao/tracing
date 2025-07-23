#include "render.cuh"

namespace cuda_device
{
    __device__ Color ray_color_device(
        const Ray &r,
        int depth,
        const Hittable *world,
        const BVHNode *bvh_nodes,
        int world_size)
    {
        if (depth <= 0)
        {
            return Color{0, 0, 0};
        }

        HitRecord rec;
        Interval ray_t{0.001f, INFINITY};

        // A more direct way without the top-level hittable dispatcher if the world IS the bvh
        if (hit_bvh(bvh_nodes, world, 0, r, ray_t, rec))
        {
            Ray scattered;
            Color attenuation;
            if (scatter(r, rec, attenuation, scattered))
            {
                return attenuation * ray_color_device(scattered, depth - 1, world, bvh_nodes, world_size);
            }
            return Color{0, 0, 0};
        }

        // If no object was hit, return the background color (sky gradient)
        Vec3 unit_direction = unit_vector(r.direction);
        float t = 0.5f * (unit_direction.y + 1.0f);
        return (1.0f - t) * Color{1.0f, 1.0f, 1.0f} + t * Color{0.5f, 0.7f, 1.0f};
    }
}

extern "C" __global__ void render_kernel(
    unsigned char *framebuffer,
    cuda_device::CameraData cam,
    const cuda_device::Hittable *world,
    const cuda_device::BVHNode *bvh_nodes,
    int world_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds to avoid writing outside the framebuffer
    if (i >= PIXEL_WIDTH || j >= PIXEL_HEIGHT)
    {
        return;
    }

    cuda_device::Color pixel_color{0, 0, 0};
    for (int s = 0; s < PIXEL_NEIGHBOR; ++s)
    {
        cuda_device::Ray r = cuda_device::get_ray_device(cam, i, j);
        pixel_color += cuda_device::ray_color_device(r, MAX_DEPTH, world, bvh_nodes, world_size);
    }

    float scale = 1.0f / PIXEL_NEIGHBOR;
    pixel_color *= scale;

    write_color(pixel_color, i, j,
                (char *)framebuffer);
}
