#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <string_view>
#include <curand_kernel.h>

extern __device__ curandState *render_rand_state_global;

// Frame rate
inline constexpr int FRAME_RATE = 30;

// Pixel format for ffmpeg
inline constexpr std::string_view PIXEL_FORMAT = "rgb24";

// Picture size
inline constexpr int PIXEL_FACTOR = 256;
inline constexpr float PIXEL_SCALE = 16.0f / 9.0f;

// Samples per pixel for anti-aliasing
inline constexpr int PIXEL_NEIGHBOR = 1;

// Tripple buffering for ffplay
inline constexpr int NUM_BUFFERS = 3;

// TODO: Configure pixel size from static file
inline constexpr int PIXEL_WIDTH = 1280;
inline constexpr int PIXEL_HEIGHT = int(float(PIXEL_WIDTH) / PIXEL_SCALE);
inline constexpr int FRAME_SIZE_RGB = PIXEL_WIDTH * PIXEL_HEIGHT * 3;

// Total ray bouncing
inline constexpr int MAX_DEPTH = 10;

// Math
inline constexpr float PI = 3.1415926535897932385;
inline constexpr float RAY_INFINITY = std::numeric_limits<float>::infinity();

// Generate random data with random distribution between 0,1
__device__ __host__ inline float random_float()
{
#ifdef __CUDA_ARCH__
    // --- DEVICE (GPU) IMPLEMENTATION ---
    // Calculate the unique global thread index
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    size_t pixel_index = j * PIXEL_WIDTH + i;

    // Use the thread's personal generator state to get a random float [0,1)
    return curand_uniform(&render_rand_state_global[pixel_index]);

#else
    // --- HOST (CPU) IMPLEMENTATION ---
    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    static std::mt19937 generator;
    return distribution(generator);

#endif
}

__device__ __host__ inline float degrees_to_radians(float degrees)
{
    return degrees * PI / 180.0f;
}

__device__ __host__ inline float random_float(float min, float max)
{
    // Returns a random real in [min,max).
    return min + (max - min) * random_float();
}

__device__ __host__ inline int random_int(int min, int max)
{
    // Returns a random integer in [min,max].
    return int(random_float(min, max + 1));
}

inline void generate_ppm_6(char *ptr)
{
    std::ofstream ppmFile("image.ppm", std::ios::out | std::ios::binary);
    if (!ppmFile.is_open())
    {
        std::cerr << "Error: could not open file image.ppm" << std::endl;
        return;
    }
    ppmFile << "P6\n"
            << PIXEL_WIDTH << " " << PIXEL_HEIGHT << "\n255\n";
    ppmFile.write(ptr, FRAME_SIZE_RGB);
    ppmFile.close();
    std::cout << "PPM file written successfully: image.ppm" << std::endl;
}

#endif // CONSTANTS_H
