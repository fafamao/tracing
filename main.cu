#include "color.h"
#include "ray.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"
#include "mem_pool.h"
#include "bvh_node.h"
#include "scene.h"
#include "random_number_generator.cuh"
#include "render.cuh"
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

        size_t new_stack_size = 16384; // 16 * 1024 bytes
        cudaDeviceSetLimit(cudaLimitStackSize, new_stack_size);
        std::cout << "Set new stack size = " << new_stack_size << " bytes\n";

        // Dimension
        int tx = 16;
        int ty = 16;

        // Allocate device memory for RNG
        size_t num_pixels = PIXEL_HEIGHT * PIXEL_WIDTH;
        curandState *scene_rand_state;
        checkCudaErrors(cudaMalloc((void **)&scene_rand_state, 1 * sizeof(curandState)));
        initialize_global_state(num_pixels);
        //  RNG kernel launch
        rand_init<<<1, 1>>>(scene_rand_state);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // Allocate device memory for scene
        size_t num_scene_objects = 22 * 22 + 4;
        int *d_object_count;
        checkCudaErrors(cudaMalloc((void **)&d_object_count, sizeof(int)));
        hittable **scene_list;
        checkCudaErrors(cudaMalloc((void **)&scene_list, num_scene_objects * sizeof(hittable *)));
        hittable **scene_world;
        checkCudaErrors(cudaMalloc((void **)&scene_world, sizeof(hittable *)));
        // Allocate device memory for camera
        Camera **d_camera;
        checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera *)));
        // Allocate device memory for pixel buffer
        char *d_pixel;
        checkCudaErrors(cudaMalloc((void **)&d_pixel, sizeof(char) * FRAME_SIZE_RGB));
        // Allocate host memory for pixel buffer
        char *h_pixel;
        checkCudaErrors(cudaMallocHost((void **)&h_pixel, sizeof(char) * FRAME_SIZE_RGB));

        // Scene kernel launch
        generate_scene_device<<<1, 1>>>(scene_list, scene_world, scene_rand_state, d_camera, d_object_count);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        //  Render our buffer
        dim3 blocks(PIXEL_WIDTH / tx + 1, PIXEL_HEIGHT / ty + 1);
        dim3 threads(tx, ty);
        render_init<<<blocks, threads>>>(PIXEL_WIDTH, PIXEL_HEIGHT);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        render_device<<<blocks, threads>>>(PIXEL_WIDTH, PIXEL_HEIGHT, d_camera, scene_world, d_pixel);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        checkCudaErrors(cudaDeviceSynchronize());
        free_scene<<<1, 1>>>(scene_list, scene_world, d_camera, d_object_count);
        checkCudaErrors(cudaDeviceSynchronize());

        cudaMemcpy(h_pixel, d_pixel, sizeof(char) * FRAME_SIZE_RGB, cudaMemcpyDeviceToHost);
        generate_ppm_6(h_pixel);

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaFree(d_camera));
        checkCudaErrors(cudaFree(scene_world));
        checkCudaErrors(cudaFree(scene_list));
        checkCudaErrors(cudaFree(scene_rand_state));
        checkCudaErrors(cudaFree(render_rand_state_global));
        checkCudaErrors(cudaFree(d_pixel));
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
