#include "camera_pod.cuh"
#include "hittable_pod.cuh"
#include "bvh_node_pod.cuh"

namespace cuda_device
{
    __device__ Color ray_color_device(
        const Ray &r,
        int depth,
        cudaTextureObject_t objects_texture,
        const BVHNode *bvh_nodes,
        int world_size);
}

#ifdef __cplusplus
extern "C"
{
#endif

    __global__ void render_kernel(
        unsigned char *framebuffer,
        cuda_device::CameraData cam,
        cudaTextureObject_t objects_texture,
        const cuda_device::BVHNode *bvh_nodes,
        int world_size);

#ifdef __cplusplus
}
#endif