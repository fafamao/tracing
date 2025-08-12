#include "camera_pod.cuh"
#include "hittable_pod.cuh"
#include "bvh_node_pod.cuh"

namespace cuda_device
{
    __device__ Color ray_color_device(
        const Ray &r,
        int depth,
        const Hittable *world,
        const BVHNode *bvh_nodes,
        int world_size);
}

#ifdef __cplusplus
extern "C"
{
#endif

    __global__ void render_kernel(
        uchar3 *framebuffer,
        cuda_device::CameraData cam,
        const cuda_device::Hittable *world,
        const cuda_device::BVHNode *bvh_nodes,
        int world_size);

#ifdef __cplusplus
}
#endif