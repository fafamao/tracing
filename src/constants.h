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
inline constexpr double PIXEL_SCALE = 16.0 / 9.0;

// Samples per pixel for anti-aliasing
inline constexpr int PIXEL_NEIGHBOR = 10;

// TODO: Configure pixel size from static file
inline constexpr int PIXEL_WIDTH = 1280;
inline constexpr int PIXEL_HEIGHT = int(double(PIXEL_WIDTH) / PIXEL_SCALE);

// Total ray bouncing
inline constexpr int MAX_DEPTH = 10;

// Math
inline constexpr double PI = 3.1415926535897932385;
inline constexpr double RAY_INFINITY = std::numeric_limits<double>::infinity();

// Generate random data with random distribution between 0,1
__device__ __host__ inline double random_double()
{
#ifdef __CUDA_ARCH__
    // --- DEVICE (GPU) IMPLEMENTATION ---
    // Calculate the unique global thread index
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    size_t pixel_index = j * PIXEL_WIDTH + i;

    // Use the thread's personal generator state to get a random double [0,1)
    return curand_uniform_double(&render_rand_state_global[pixel_index]);

#else
    // --- HOST (CPU) IMPLEMENTATION ---
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);

#endif
}

__device__ __host__ inline double degrees_to_radians(double degrees)
{
    return degrees * PI / 180.0;
}

__device__ __host__ inline double random_double(double min, double max)
{
    // Returns a random real in [min,max).
    return min + (max - min) * random_double();
}

__device__ __host__ inline int random_int(int min, int max)
{
    // Returns a random integer in [min,max].
    return int(random_double(min, max + 1));
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
    ppmFile.write(ptr, PIXEL_HEIGHT * PIXEL_WIDTH * 3);
    ppmFile.close();
    std::cout << "PPM file written successfully: image.ppm" << std::endl;
}

#endif // CONSTANTS_H
