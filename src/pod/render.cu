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
        Color final_color{0.0f, 0.0f, 0.0f};
        Color throughput{1.0f, 1.0f, 1.0f};
        Ray current_ray = r;

        for (int i = 0; i < depth; ++i)
        {
            HitRecord rec;
            Interval ray_t{0.001f, INFINITY};

            bool hit = hit_bvh(bvh_nodes, world, 0, current_ray, ray_t, rec);
            float hit_mask = static_cast<float>(hit);

            // Outcome A: What happens on a HIT
            Ray scattered;
            Color attenuation;
            bool did_scatter = scatter(current_ray, rec, attenuation, scattered);
            Color hit_throughput = throughput * attenuation * static_cast<float>(did_scatter);
            Ray hit_ray = scattered;

            // Outcome B: What happens on a MISS (background color)
            Vec3 unit_direction = unit_vector(current_ray.direction);
            float t = 0.5f * (unit_direction.y + 1.0f);
            Color background_color = (1.0f - t) * Color{1.0f, 1.0f, 1.0f} + t * Color{0.5f, 0.7f, 1.0f};
            Color miss_contribution = throughput * background_color;

            // 3. Blend the results using the hit_mask

            // a) Update the final color. Add the miss contribution only if there was no hit.
            final_color += (1.0f - hit_mask) * miss_contribution;

            // b) Update the ray and throughput for the next iteration.
            // If there was a hit, use the hit values. If a miss, the values don't matter
            // because the loop will terminate.
            throughput = hit_mask * hit_throughput;
            current_ray = hit_mask * hit_ray;

            // 4. Terminate the loop if we missed OR if the throughput is zero.
            if (!hit || near_zero(throughput))
            {
                break;
            }
        }

        return final_color;
    }
}

extern "C" __global__ void render_kernel(
    uchar3 *framebuffer,
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

    const float scale = 1.0f / PIXEL_NEIGHBOR;
    cuda_device::Color pixel_color{0.0f, 0.0f, 0.0f};
    for (int s = 0; s < PIXEL_NEIGHBOR; ++s)
    {
        cuda_device::Ray r = cuda_device::get_ray_device(cam, i, j);
        cuda_device::Color contrib = cuda_device::ray_color_device(r, MAX_DEPTH, world, bvh_nodes, world_size);
        pixel_color.x = fmaf(contrib.x, scale, pixel_color.x);
        pixel_color.y = fmaf(contrib.y, scale, pixel_color.y);
        pixel_color.z = fmaf(contrib.z, scale, pixel_color.z);
    }

    write_color(pixel_color, i, j, framebuffer);
}
