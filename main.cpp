#include "color.h"

using namespace color;

int main()
{
    for (int j = 0; j < PIXEL_HEIGHT; j++)
    {
        std::clog << "\rScanlines remaining: " << (PIXEL_HEIGHT - j) << ' ' << std::flush;
        for (int i = 0; i < PIXEL_WIDTH; i++)
        {
            auto pixel_color = Color(double(i) / (PIXEL_WIDTH - 1), double(j) / (PIXEL_HEIGHT - 1), 0);
            pixel_color.display_color();
        }
    }
    return 0;
}
