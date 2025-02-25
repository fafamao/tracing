#include "color.h"
#include "ray.h"

using namespace color;
using namespace ray;

double hit_sphere(const Vec3 &center, double radius, const Ray &r)
{
    Vec3 oc = center - r.get_origin();
    auto a = dot(r.get_direction(), r.get_direction());
    // To simplify the square root calculation with b = -2h
    // auto b = -2.0 * dot(r.get_direction(), oc);
    auto h = dot(r.get_direction(), oc);
    auto c = dot(oc, oc) - radius * radius;
    auto discriminant = h * h - a * c;
    if (discriminant < 0)
    {
        return -1.0;
    }
    else
    {
        return (h - std::sqrt(discriminant)) / a;
    }
}

Color ray_color(const Ray &r)
{
    auto t = hit_sphere(Vec3(0, 0, -1), 0.5, r);
    if (t > 0.0)
    {
        Vec3 N = unit_vector(r.at(t) - Vec3(0, 0, -1));
        return 0.5 * Color(N.get_x() + 1, N.get_y() + 1, N.get_z() + 1);
    }
    Vec3 unit_direction = unit_vector(r.get_direction());
    auto a = 0.5 * (unit_direction.get_y() + 1.0);
    return (1.0 - a) * Color(1.0, 1.0, 1.0) + a * Color(0.5, 0.7, 1.0);
}

int main()
{
    // Camera is looking at the center of the picture
    auto camera = Vec3(0, 0, 0);
    // Horizontal vector and vertical vector
    auto vec_u = Vec3(VIEWPORT_WIDTH, 0, 0);
    auto vec_v = Vec3(0, -VIEWPORT_HEIGHT, 0);
    // Unit vector between pixels
    auto unit_vec_u = vec_u / PIXEL_WIDTH;
    auto unit_vec_v = vec_v / PIXEL_HEIGHT;
    // Position of the top left pixel
    auto top_left_pixel = camera - Vec3(0, 0, FOCAL_LEN) - vec_u / 2 - vec_v / 2 + unit_vec_u / 2 + unit_vec_v / 2;

    std::cout << "P3\n"
              << PIXEL_WIDTH << ' ' << PIXEL_HEIGHT << "\n255\n";

    for (int j = 0; j < PIXEL_HEIGHT; j++)
    {
        std::clog << "\rScanlines remaining: " << (PIXEL_HEIGHT - j) << ' ' << std::flush;
        for (int i = 0; i < PIXEL_WIDTH; i++)
        {
            auto current_pixel = top_left_pixel + j * unit_vec_v + i * unit_vec_u;
            auto ray_direction = current_pixel - camera;
            Ray r(camera, ray_direction);
            auto pixel_color = ray_color(r);
            pixel_color.display_color();
        }
    }
    return 0;
}
