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

    inline double linear_to_gamma(double linear_component)
    {
        if (linear_component > 0)
            return std::sqrt(linear_component);

        return 0;
    }

public:
    Color() {};
    Color(double r1, double g1, double b1) : r(r1), g(g1), b(b1) {};
    Color(const Color &color)
    {
        r = color.r;
        g = color.g;
        b = color.b;
    }

    // Method to generate pixel
    void display_color()
    {
        // Apply a linear to gamma transform for gamma 2
        r = linear_to_gamma(r);
        g = linear_to_gamma(g);
        b = linear_to_gamma(b);

        int rp = int(PIXEL_FACTOR * pixel_interval.clamp(r));
        int gp = int(PIXEL_FACTOR * pixel_interval.clamp(g));
        int bp = int(PIXEL_FACTOR * pixel_interval.clamp(b));
        std::cout << rp << ' ' << gp << ' ' << bp << '\n';
    }

    double get_r() const { return r; }
    double get_g() const { return g; }
    double get_b() const { return b; }

    Color &operator-=(const Color &color2)
    {
        r = r - color2.get_r();
        g = g - color2.get_g();
        b = b - color2.get_b();
        return *this;
    }
    Color &operator+=(const Color &color2)
    {
        r = r + color2.get_r();
        g = g + color2.get_g();
        b = b + color2.get_b();
        return *this;
    }
    Color &operator*=(const Color &color2)
    {
        r = r * color2.get_r();
        g = g * color2.get_g();
        b = b * color2.get_b();
        return *this;
    }
    // Operator to calculate color when the ray hit the surface
    Color operator+(const Vec3 &normal)
    {
        return Color(r + normal.get_x(), g + normal.get_y(), b + normal.get_z());
    }
    // Operator to scale self
    Color &operator*=(const double scale)
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
inline Color operator*(const double t, const Color &color)
{
    return Color(t * color.get_r(), t * color.get_g(), t * color.get_b());
}
// Operator to add
inline Color operator+(const Color &color1, const Color &color2)
{
    return Color(color1.get_r() + color2.get_r(), color1.get_g() + color2.get_g(), color1.get_b() + color2.get_b());
}
// Operator to minus
inline Color operator-(const Color &color1, const Color &color2)
{
    return Color(color1.get_r() - color2.get_r(), color1.get_g() - color2.get_g(), color1.get_b() - color2.get_b());
}

#endif // COLOR_H
