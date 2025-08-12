#ifndef COLOR_POD_CUH_
#define COLOR_POD_CUH_

#include "constants.h"
#include "interval_pod.cuh"
#include "vec3_pod.cuh"
#include <iostream>

namespace cuda_device
{

  using Color = Vec3;

  // Gamma correction helper
  __device__ inline float linear_to_gamma(float linear_component)
  {
    return sqrtf(fmaxf(0.0f, linear_component));
  }

  // Writes the final color to the pixel buffer
  __device__ inline void write_color(Color &pixel_color, int i, int j,
                                     uchar3 *ptr)
  {
    float r = linear_to_gamma(pixel_color.x);
    float g = linear_to_gamma(pixel_color.y);
    float b = linear_to_gamma(pixel_color.z);

    // Get the clamp interval
    unsigned char rp = static_cast<unsigned char>(255.999f * interval_clamp(PIXEL_INTENSITY_INTERVAL, r));
    unsigned char gp = static_cast<unsigned char>(255.999f * interval_clamp(PIXEL_INTENSITY_INTERVAL, g));
    unsigned char bp = static_cast<unsigned char>(255.999f * interval_clamp(PIXEL_INTENSITY_INTERVAL, b));

    ptr[j * PIXEL_WIDTH + i] = make_uchar3(rp, gp, bp);
  }

  // Generates a random color
  __device__ inline Color random_color()
  {
    return Color{random_float(), random_float(), random_float()};
  }

  __device__ inline Color random_color(float min, float max)
  {
    return Color{random_float(min, max), random_float(min, max),
                 random_float(min, max)};
  }
}
#endif // COLOR_POD_CUH_