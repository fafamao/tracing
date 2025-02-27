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
            auto current_pixel = _top_left_pixel + j * _unit_vec_v + i * _unit_vec_u;
            auto ray_direction = current_pixel - _camera;
            Ray r(_camera, ray_direction);
            auto pixel_color = ray_color(r, world);
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
}

Color Camera::ray_color(const Ray &r, const hittable_list &world)
{
    hit_record record;
    Interval interval(0, RAY_INFINITY);
    // Check if ray hits the hittable list and save record
    if (world.hit(r, interval, record))
    {
        return 0.5 * (Color(1, 1, 1) + record.normal);
    }
    Vec3 unit_direction = unit_vector(r.get_direction());
    auto a = 0.5 * (unit_direction.get_y() + 1.0);
    return (1.0 - a) * Color(1.0, 1.0, 1.0) + a * Color(0.5, 0.7, 1.0);
}