#ifndef INTERVAL_H_
#define INTERVAL_H_

#include "constants.h"

class Interval
{
public:
    double min, max;

    Interval() : min(RAY_INFINITY), max(-RAY_INFINITY) {} // Default interval is empty

    Interval(double min, double max) : min(min), max(max) {}

    Interval(Interval a, Interval b)
    {
        min = a.min < b.min ? a.min : b.min;
        max = a.max > b.max ? a.max : b.max;
    }

    double size() const
    {
        return max - min;
    }

    bool contains(double x) const
    {
        return min <= x && x <= max;
    }

    bool surrounds(double x) const
    {
        return min < x && x < max;
    }

    double clamp(double x) const
    {
        if (x < min)
            return min;
        if (x > max)
            return max;
        return x;
    }

    static const Interval empty, universe;
};

static Interval pixel_interval(0.000, 0.999);

#endif // INTERVAL_H_