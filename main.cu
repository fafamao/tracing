#include "legacy/color.h"
#include "legacy/ray.h"
#include "legacy/hittable_list.h"
#include "legacy/sphere.h"
#include "legacy/camera.h"
#include "legacy/material.h"
#include "utility/mem_pool.h"
#include "legacy/bvh_node.h"
#include "legacy/scene.h"
#include "random_number_generator.cuh"
#include "legacy/render.cuh"
#include "pod/generate_scene.h"
#include "pod/bvh_builder.cuh"
#include <cuda_runtime.h>
#include <cstring>

// Global random state
__device__ curandState *render_rand_state_global;

bool is_gpu_available()
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("GPU card is not available.\n");
        return false;
    }
    return deviceCount > 0;
}

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

void initialize_global_state(int num_pixels)
{
    curandState *d_temp_ptr;
    checkCudaErrors(cudaMalloc((void **)&d_temp_ptr, num_pixels * sizeof(curandState)));
    checkCudaErrors(cudaMemcpyToSymbol(render_rand_state_global, &d_temp_ptr, sizeof(d_temp_ptr)));
}

int main()
{
    // Check GPU availability
    bool is_gpu_ready = is_gpu_available();

    if (is_gpu_ready)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        printf("Device %d: %s\n", 0, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024 * 1024));
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Warp Size: %d\n", prop.warpSize);

        size_t current_stack_size;
        cudaDeviceGetLimit(&current_stack_size, cudaLimitStackSize);
        std::cout << "Current stack size = " << current_stack_size << " bytes\n";

        // Dimension
        int tx = 16;
        int ty = 16;

        size_t new_stack_size = 16384; // 16 * 1024 bytes
        cudaDeviceSetLimit(cudaLimitStackSize, new_stack_size);
        std::cout << "Set new stack size = " << new_stack_size << " bytes\n";

        // Construct scene data
        std::vector<cuda_device::Hittable> world_hittable = cuda_device::generate_world();
        cuda_device::BVHBuilder bvh_node(world_hittable);
        std::vector<cuda_device::BVHNode> bvh_nodes = bvh_node.build();
        // Copy scene data to device
        cuda_device::Hittable *d_objects;
        cuda_device::BVHNode *d_nodes;
        checkCudaErrors(cudaMalloc(&d_objects, world_hittable.size() * sizeof(cuda_device::Hittable)));
        checkCudaErrors(cudaMalloc(&d_nodes, bvh_nodes.size() * sizeof(cuda_device::BVHNode)));
        cudaMemcpy(d_objects, world_hittable.data(), world_hittable.size() * sizeof(cuda_device::Hittable), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nodes, bvh_nodes.data(), bvh_nodes.size() * sizeof(cuda_device::BVHNode), cudaMemcpyHostToDevice);

        // Initialize random number of each pixel
        render_init<<<1, 1>>>(PIXEL_WIDTH, PIXEL_HEIGHT);
        checkCudaErrors(cudaDeviceSynchronize());
    }
    else
    {
        // Instantiate thread pool
        ThreadPool tp;
        // Memory pool to hold rgb values
        size_t rgb_size = PIXEL_HEIGHT * PIXEL_WIDTH * 3;
        size_t pool_siz = rgb_size * 2;
        MemoryPool mem_pool(pool_siz);
        char *pixel_buffer = mem_pool.allocate(rgb_size);

        // Initialize pixel_buffer all white
        memset(pixel_buffer, 255, rgb_size);

        // Create scene
        hittable_list world;
        generate_scene_host(world);
        // Create camera
        Vec3 camera_origin = Vec3(13, 2, 3);
        Vec3 camera_dest = Vec3(0, 0, 0);
        Vec3 camera_up = Vec3(0, 1, 0);
        Camera camera(camera_origin, camera_dest, camera_up, &tp);
        // Start rendering
        camera.render(world, pixel_buffer);
    }

    printf("Rendering ends\n");

    return 0;
}
