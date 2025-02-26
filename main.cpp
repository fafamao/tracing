#include "color.h"
#include "ray.h"
#include "hittable_list.h"
#include "sphere.h"

using namespace color;
using namespace ray;


Color ray_color(const Ray &r, const hittable_list& world)
{
    hit_record record;
    // Check if ray hits the hittable list and save record
    if (world.hit(r, 0, INFINITY, record))
    {
        return 0.5 * (record.normal + Color(1,1,1));
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

    // Add earth and universe
    hittable_list world;
    world.add(make_shared<sphere>(Vec3(0,0,-1), 0.5));
    world.add(make_shared<sphere>(Vec3(0,-100.5,-1), 100));


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
            auto pixel_color = ray_color(r, world);
            pixel_color.display_color();
        }
    }
    return 0;
}
