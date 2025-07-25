#include "interval_pod.cuh"

namespace cuda_device
{
    const Interval EMPTY_INTERVAL = {RAY_INFINITY, -RAY_INFINITY};
    // Represents the entire number line.
    const Interval UNIVERSE_INTERVAL = {-RAY_INFINITY, RAY_INFINITY};
    // The interval for pixel intensity.
    //__constant__ Interval PIXEL_INTENSITY_INTERVAL = {0.000f, 0.999f};
}