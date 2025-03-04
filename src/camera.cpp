#include "camera.h"

void Camera::render(const hittable_list &world)
{
    std::cout << "P3\n"
              << PIXEL_WIDTH << ' ' << PIXEL_HEIGHT << "\n255\n";

    for (int j = 0; j < PIXEL_HEIGHT; j++)
    {
        std::clog << "\rScanlines remaining: " << (PIXEL_HEIGHT - j) << ' ' << std::flush;
        for (int i = 0; i < PIXEL_WIDTH; i++)
        {
            Color pixel_color(0, 0, 0);
            for (int sample = 0; sample < PIXEL_NEIGHBOR; sample++)
            {
                Ray r = get_ray(i, j);
                pixel_color += ray_color(r, MAX_DEPTH, world);
            }
            pixel_color *= _pixel_scale;
            pixel_color.display_color();
        }
    }
}

void Camera::initialize()
{
    // Camera is placed at the origin of the axis
    _camera = Vec3(0, 0, 0);
    // Horizontal vector and vertical vector
    auto vec_u = Vec3(VIEWPORT_WIDTH, 0, 0);
    auto vec_v = Vec3(0, -VIEWPORT_HEIGHT, 0);
    // Unit vector between pixels
    _unit_vec_u = vec_u / PIXEL_WIDTH;
    _unit_vec_v = vec_v / PIXEL_HEIGHT;
    // Position of the top left pixel
    _top_left_pixel = _camera - Vec3(0, 0, FOCAL_LEN) - vec_u / 2 - vec_v / 2 + _unit_vec_u / 2 + _unit_vec_v / 2;
    // Color scale factor for a number of samples
    _pixel_scale = 1.0 / PIXEL_NEIGHBOR;
}

Color Camera::ray_color(const Ray &r, const int depth, const hittable_list &world)
{
    if (depth <= 0)
        return Color(0, 0, 0);
    hit_record record;
    // Avoid floating point error to have 0.001
    Interval interval(0.001, RAY_INFINITY);
    // Check if ray hits the hittable list and save record
    if (world.hit(r, interval, record))
    {
        Ray scattered;
        Color attenuation;
        if (record.mat->scatter(r, record.normal, record.p, attenuation, scattered))
            return ray_color(scattered, depth - 1, world) *= attenuation;
        return Color(0, 0, 0);
    }
    Vec3 unit_direction = unit_vector(r.get_direction());
    auto a = 0.5 * (unit_direction.get_y() + 1.0);
    return (1.0 - a) * Color(1.0, 1.0, 1.0) + a * Color(0.5, 0.7, 1.0);
}