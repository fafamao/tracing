#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cmath>
#include <random>

// Picture size
inline constexpr int PIXEL_FACTOR = 256;
inline constexpr double PIXEL_SCALE = 16.0 / 9.0;

// Samples per pixel for anti-aliasing
inline constexpr int PIXEL_NEIGHBOR = 10;

// Focal length
inline constexpr double FOCAL_LEN = 1.0;

// TODO: Configure pixel size from static file
inline constexpr int PIXEL_WIDTH = 400;
inline constexpr int PIXEL_HEIGHT = int(double(PIXEL_WIDTH) / PIXEL_SCALE);

// Total ray bouncing
inline constexpr int MAX_DEPTH = 10;

// Viewport size
/* inline constexpr double VIEWPORT_WIDTH = 4.8;
inline constexpr double VIEWPORT_HEIGHT = VIEWPORT_WIDTH / (PIXEL_WIDTH / PIXEL_HEIGHT); */
inline constexpr double VIEWPORT_HEIGHT = 2.0;
inline constexpr double VIEWPORT_WIDTH = VIEWPORT_HEIGHT * (double(PIXEL_WIDTH) / double(PIXEL_HEIGHT));

// Math
inline constexpr double PI = 3.1415926535897932385;
inline constexpr double RAY_INFINITY = std::numeric_limits<double>::infinity();

// Generate random data with random distribution between 0,1
inline double random_double()
{
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

inline double random_double(double min, double max)
{
    // Returns a random real in [min,max).
    return min + (max - min) * random_double();
}

#endif // CONSTANTS_H
