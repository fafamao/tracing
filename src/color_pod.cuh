#ifndef COLOR_POD_CUH_
#define COLOR_POD_CUH_

#include "vec3_pod.cuh"
#include "constants.h"
#include "interval_pod.cuh"
#include <iostream>

using Color = Vec3;

// Gamma correction helper
__device__ inline float linear_to_gamma(float linear_component)
{
    return sqrtf(fmaxf(0.0f, linear_component));
}

// Writes the final color to the pixel buffer
__device__ inline void write_color(Color &pixel_color, int i, int j, char *const ptr)
{
    float r = linear_to_gamma(pixel_color.x);
    float g = linear_to_gamma(pixel_color.y);
    float b = linear_to_gamma(pixel_color.z);

    // Get the clamp interval
    unsigned char rp = static_cast<unsigned char>(256 * interval_clamp(PIXEL_INTENSITY_INTERVAL, r));
    unsigned char gp = static_cast<unsigned char>(256 * interval_clamp(PIXEL_INTENSITY_INTERVAL, g));
    unsigned char bp = static_cast<unsigned char>(256 * interval_clamp(PIXEL_INTENSITY_INTERVAL, b));

    // Calculate buffer position
    int buff_pos = (j * PIXEL_WIDTH + i) * 3;

    ptr[buff_pos] = rp;
    ptr[buff_pos + 1] = gp;
    ptr[buff_pos + 2] = bp;
}

// Generates a random color
__device__ inline Color random_color()
{
    return Color{random_float(), random_float(), random_float()};
}

__device__ inline Color random_color(float min, float max)
{
    return Color{random_float(min, max), random_float(min, max), random_float(min, max)};
}

// Operator to add a Vec3 (like a normal) to a Color
__device__ inline Color operator+(const Color &color, const Vec3 &vec)
{
    return Color{color.x + vec.x, color.y + vec.y, color.z + vec.z};
}

#endif // COLOR_POD_CUH_