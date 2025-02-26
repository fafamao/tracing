#ifndef COLOR_H
#define COLOR_H

#include <iostream>
#include "constants.h"

namespace color
{
    class Color
    {
    private:
        double r, g, b;

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
            int rp = int(PIXEL_FACTOR * r);
            int gp = int(PIXEL_FACTOR * g);
            int bp = int(PIXEL_FACTOR * b);
            std::cout << rp << ' ' << gp << ' ' << bp << '\n';
        }

        double get_r() const {return r;}
        double get_g() const {return g;}
        double get_b() const {return b;}

        Color& operator-=(const Color& color2) {
            r = r - color2.get_r();
            g = g - color2.get_g();
            b = b - color2.get_b();
            return *this;
        }
        Color& operator+=(const Color& color2) {
            r = r + color2.get_r();
            g = g + color2.get_g();
            b = b + color2.get_b();
            return *this;
        }
        Color& operator*=(const Color& color2) {
            r = r * color2.get_r();
            g = g * color2.get_g();
            b = b * color2.get_b();
            return *this;
        }

    };
    // Operator to scale Color
    inline Color operator*(const double t, const Color& color) {
        return Color(t * color.get_r(), t * color.get_g(), t * color.get_b());
    }
    // Operator to add
    inline Color operator+(const Color& color1, const Color& color2) {
        return Color(color1.get_r() * color2.get_r(), color1.get_g() * color2.get_g(), color1.get_b() * color2.get_b());
    }
    // Operator to minus
    inline Color operator-(const Color& color1, const Color& color2) {
        return Color(color1.get_r() - color2.get_r(), color1.get_g() - color2.get_g(), color1.get_b() - color2.get_b());
    }
}

#endif // COLOR_H
