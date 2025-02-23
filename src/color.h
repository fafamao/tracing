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
    };
}

#endif // COLOR_H
