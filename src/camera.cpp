#include "camera.h"

void Camera::render(const hittable_list &world, char* ptr)
{
    for (int j = 0; j < PIXEL_HEIGHT; j++)
    {
        for (int i = 0; i < PIXEL_WIDTH; i++)
        {
            Color pixel_color(0, 0, 0);
            std::function<void()> rendering = [&i, &j, &world, ptr, &pixel_color, this]()
            {
                for (int sample = 0; sample < PIXEL_NEIGHBOR; sample++)
                {

                    Ray r = get_ray(i, j);
                    pixel_color += ray_color(r, MAX_DEPTH, world);
                }
                pixel_color *= _pixel_scale;
                pixel_color.write_color(i, j, ptr);
            };
            thread_pool.enqueue(rendering);
        }
    }

    //generate_ppm_6(ptr);
}

void Camera::redering_done() {

}

void Camera::initialize()
{
    // Camera is placed at the origin of the axis
    _camera = lookfrom;
    // Determine focal length
    auto focal_len = (lookfrom - lookat).get_length();
    auto theta = degrees_to_radians(vfov);
    auto h = std::tan(theta / 2);
    auto viewport_height = 2 * h * focal_len;
    auto viewport_width = viewport_height * (double(PIXEL_WIDTH) / double(PIXEL_HEIGHT));
    // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);
    // Horizontal vector and vertical vector
    auto vec_u = u * viewport_width;
    auto vec_v = -v * viewport_height;
    // Unit vector between pixels
    _unit_vec_u = vec_u / PIXEL_WIDTH;
    _unit_vec_v = vec_v / PIXEL_HEIGHT;
    // Position of the top left pixel
    _top_left_pixel = _camera - focal_len * w - vec_u / 2 - vec_v / 2 + _unit_vec_u / 2 + _unit_vec_v / 2;
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
        // Workaround to avoid circular dependency, construct a struct to replace record
        record_content temp_record;
        temp_record.front_face = record.front_face;
        temp_record.normal = record.normal;
        temp_record.p = record.p;
        Ray scattered;
        Color attenuation;
        if (record.mat->scatter(r, temp_record, attenuation, scattered))
            return ray_color(scattered, depth - 1, world) *= attenuation;
        return Color(0, 0, 0);
    }
    Vec3 unit_direction = unit_vector(r.get_direction());
    auto a = 0.5 * (unit_direction.get_y() + 1.0);
    return (1.0 - a) * Color(1.0, 1.0, 1.0) + a * Color(0.5, 0.7, 1.0);
}