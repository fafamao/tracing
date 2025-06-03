#ifndef INTERVAL_H_
#define INTERVAL_H_

#include "constants.h"

class Interval
{
public:
    double min, max;

    __host__ __device__ Interval() : min(RAY_INFINITY), max(-RAY_INFINITY) {} // Default interval is empty

    __host__ __device__ Interval(double min, double max) : min(min), max(max) {}

    __host__ __device__ Interval(Interval a, Interval b)
    {
        min = a.min < b.min ? a.min : b.min;
        max = a.max > b.max ? a.max : b.max;
    }

    __host__ __device__ double size() const
    {
        return max - min;
    }

    __host__ __device__ bool contains(double x) const
    {
        return min <= x && x <= max;
    }

    __host__ __device__ bool surrounds(double x) const
    {
        return min < x && x < max;
    }

    __host__ __device__ double clamp(double x) const
    {
        if (x < min)
            return min;
        if (x > max)
            return max;
        return x;
    }

};

__host__ __device__ static Interval get_pixel_interval() {
    return Interval(0.000, 0.999);
}

#endif // INTERVAL_H_