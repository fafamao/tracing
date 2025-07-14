#ifndef INTERVAL_POD_CUH_
#define INTERVAL_POD_CUH_

#include "constants.h"
namespace cuda_device
{

    // The POD struct only contains the data.
    struct Interval
    {
        float min, max;
    };

    // Represents an empty interval.
    const Interval EMPTY_INTERVAL = {RAY_INFINITY, -RAY_INFINITY};
    // Represents the entire number line.
    const Interval UNIVERSE_INTERVAL = {-RAY_INFINITY, RAY_INFINITY};
    // The interval for pixel intensity.
    const Interval PIXEL_INTENSITY_INTERVAL = {0.000f, 0.999f};

    // Calculates the size of the interval.
    __device__ inline float interval_size(const Interval &interval)
    {
        return interval.max - interval.min;
    }

    // Checks if the interval contains a value (inclusive).
    __device__ inline bool interval_contains(const Interval &interval, float x)
    {
        return interval.min <= x && x <= interval.max;
    }

    // Checks if the interval surrounds a value (exclusive).
    __device__ inline bool interval_surrounds(const Interval &interval, float x)
    {
        return interval.min < x && x < interval.max;
    }

    // Clamps a value to the interval.
    __device__ inline float interval_clamp(const Interval &interval, float x)
    {
        if (x < interval.min)
            return interval.min;
        if (x > interval.max)
            return interval.max;
        return x;
    }

    // Creates a new interval that contains two other intervals.
    __device__ inline Interval create_combined_interval(const Interval &a, const Interval &b)
    {
        float new_min = a.min < b.min ? a.min : b.min;
        float new_max = a.max > b.max ? a.max : b.max;
        return Interval{new_min, new_max};
    }
}
#endif // INTERVAL_POD_CUH_