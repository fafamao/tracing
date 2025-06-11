#include "color.h"
#include "ray.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"
#include "mem_pool.h"
#include "bvh_node.h"
#include "scene.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstring>

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

__global__ void rand_init(curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_index = j * max_x + i;
    // Same id and different seed boosts performance
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
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

        // Dimension
        int tx = 8;
        int ty = 8;

        // RNG
        size_t num_pixels = PIXEL_HEIGHT * PIXEL_WIDTH;
        curandState *d_rand_state;
        checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
        curandState *d_rand_state2;
        checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(curandState)));

        // we need that 2nd random state to be initialized for the world creation
        rand_init<<<1, 1>>>(d_rand_state2);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        clock_t start, stop;
        start = clock();
        // Render our buffer
        dim3 blocks(PIXEL_WIDTH / tx + 1, PIXEL_HEIGHT / ty + 1);
        dim3 threads(tx, ty);
        render_init<<<blocks, threads>>>(PIXEL_WIDTH, PIXEL_HEIGHT, d_rand_state);
        checkCudaErrors(cudaGetLastError());
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
