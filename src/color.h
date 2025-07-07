#ifndef COLOR_H
#define COLOR_H

#include <iostream>
#include "vec3.h"
#include "constants.h"
#include "interval.h"

class Color
{
private:
    double r, g, b;

    __host__ __device__ inline double linear_to_gamma(double linear_component)
    {
#ifdef CUDA_ARCH
        return sqrtf(fmaxf(0.0f, linear_component));
#else
        return std::sqrt(std::fmax(0.0f, linear_component));
#endif
    }

public:
    __host__ __device__ Color() {};
    __host__ __device__ Color(double r1, double g1, double b1) : r(r1), g(g1), b(b1) {};
    __host__ __device__ Color(const Color &color)
    {
        r = color.r;
        g = color.g;
        b = color.b;
    }

    // Write color to memory
    __host__ __device__ void write_color(int i, int j, char *const ptr)
    {
        r = linear_to_gamma(r);
        g = linear_to_gamma(g);
        b = linear_to_gamma(b);

        unsigned char rp = static_cast<unsigned char>(PIXEL_FACTOR * get_pixel_interval().clamp(r));
        unsigned char gp = static_cast<unsigned char>(PIXEL_FACTOR * get_pixel_interval().clamp(g));
        unsigned char bp = static_cast<unsigned char>(PIXEL_FACTOR * get_pixel_interval().clamp(b));
        // 3 for r,g,b
        int buff_pos = (i + j * PIXEL_WIDTH) * 3;

        *(ptr + buff_pos) = rp;
        *(ptr + (buff_pos + 1)) = gp;
        *(ptr + (buff_pos + 2)) = bp;
    }

    __host__ __device__ double get_r() const { return r; }
    __host__ __device__ double get_g() const { return g; }
    __host__ __device__ double get_b() const { return b; }

    __host__ __device__ Color &operator-=(const Color &color2)
    {
        r = r - color2.get_r();
        g = g - color2.get_g();
        b = b - color2.get_b();
        return *this;
    }
    __host__ __device__ Color &operator+=(const Color &color2)
    {
        r = r + color2.get_r();
        g = g + color2.get_g();
        b = b + color2.get_b();
        return *this;
    }
    __host__ __device__ Color &operator*=(const Color &color2)
    {
        r = r * color2.get_r();
        g = g * color2.get_g();
        b = b * color2.get_b();
        return *this;
    }
    // Operator to calculate color when the ray hit the surface
    __host__ __device__ Color operator+(const Vec3 &normal)
    {
        return Color(r + normal.get_x(), g + normal.get_y(), b + normal.get_z());
    }
    // Operator to scale self
    __host__ __device__ Color &operator*=(const double scale)
    {
        r = r * scale;
        g = g * scale;
        b = b * scale;
        return *this;
    }
    static Color random_color()
    {
        return Color(random_double(), random_double(), random_double());
    }
    static Color random_color(double min, double max)
    {
        return Color(random_double(min, max), random_double(min, max), random_double(min, max));
    }
};
// Operator to scale Color
__host__ __device__ inline Color operator*(const double t, const Color &color)
{
    return Color(t * color.get_r(), t * color.get_g(), t * color.get_b());
}
// Operator to add
__host__ __device__ inline Color operator+(const Color &color1, const Color &color2)
{
    return Color(color1.get_r() + color2.get_r(), color1.get_g() + color2.get_g(), color1.get_b() + color2.get_b());
}
// Operator to minus
__host__ __device__ inline Color operator-(const Color &color1, const Color &color2)
{
    return Color(color1.get_r() - color2.get_r(), color1.get_g() - color2.get_g(), color1.get_b() - color2.get_b());
}

#endif // COLOR_H
