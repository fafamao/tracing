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

int main()
{
    // Check GPU availability
    bool is_gpu_ready = is_gpu_available();

    // Instantiate memory pool
    // TODO: fix this
    size_t rgb_size = PIXEL_HEIGHT * PIXEL_WIDTH * 3;
    size_t pool_siz = rgb_size * 2;
    MemoryPool mem_pool(pool_siz);
    char *pixel_buffer = mem_pool.allocate(rgb_size);

    // Initialize pixel_buffer all white
    memset(pixel_buffer, 255, rgb_size);

    /*     // --- Construct the ffplay command ---
        std::string cmd = "ffplay ";
        cmd += "-v error "; // Quieter log level (shows errors only)
        // Input options - must match the data being piped
        cmd += "-f rawvideo ";
        cmd += "-pixel_format rgb24 "; // TODO: define format in constants.h
        cmd += "-video_size " + std::to_string(PIXEL_WIDTH) + "x" + std::to_string(PIXEL_HEIGHT) + " ";
        cmd += "-framerate " + std::to_string(FRAME_RATE) + " ";
        // Other options
        cmd += "-window_title \"Ray Tracer Output (ffplay)\" "; // Optional: Set window title
        cmd += "-i - ";                                         // Read input from stdin

        std::cout << "Executing command: " << cmd << std::endl;

        FILE *pipe = nullptr;

        pipe = popen(cmd.c_str(), "w");
        if (!pipe)
        {
            throw std::runtime_error("popen() to ffplay failed! Is ffplay installed and in PATH?");
        }
        std::cout << "ffplay process started. Rendering frames..." << std::endl;

        if (pipe)
        {
            pclose(pipe);
        } */

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

        hittable_list* world = generate_scene_device();
    }
    else
    {
        // Instantiate thread pool
        ThreadPool tp;
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
