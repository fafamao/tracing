#ifndef INTERVAL_H_
#define INTERVAL_H_

#include "constants.h"

class Interval
{
public:
    float min, max;

    __host__ __device__ Interval() : min(RAY_INFINITY), max(-RAY_INFINITY) {} // Default interval is empty

    __host__ __device__ Interval(float min, float max) : min(min), max(max) {}

    __host__ __device__ Interval(Interval a, Interval b)
    {
        min = a.min < b.min ? a.min : b.min;
        max = a.max > b.max ? a.max : b.max;
    }

    __host__ __device__ float size() const
    {
        return max - min;
    }

    __host__ __device__ bool contains(float x) const
    {
        return min <= x && x <= max;
    }

    __host__ __device__ bool surrounds(float x) const
    {
        return min < x && x < max;
    }

    __host__ __device__ float clamp(float x) const
    {
        if (x < min)
            return min;
        if (x > max)
            return max;
        return x;
    }
};

__host__ __device__ static Interval get_pixel_interval()
{
    return Interval(0.000f, 0.999f);
}

#endif // INTERVAL_H_