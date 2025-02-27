#ifndef INTERVAL_H_
#define INTERVAL_H_

#include "constants.h"

class Interval
{
public:
    double min, max;

    Interval() : min(RAY_INFINITY), max(-RAY_INFINITY) {} // Default interval is empty

    Interval(double min, double max) : min(min), max(max) {}

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

    static const Interval empty, universe;
};

#endif // INTERVAL_H_