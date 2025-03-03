#ifndef CAMERA_H_
#define CAMERA_H_

#include "vec3.h"
#include "constants.h"
#include "color.h"
#include "hittable_list.h"

class Camera
{
private:
    Vec3 _top_left_pixel, _unit_vec_v, _unit_vec_u, _camera;
    double _pixel_scale;
    void initialize();
    Color ray_color(const Ray &r, const hittable_list &world);

public:
    Camera()
    {
        initialize();
    };
    void render(const hittable_list &world);

    // Generate vectors pointing to ([-0.5,0.5], [-0.5, 0.5], 0)
    Vec3 sample_square() const {
        return Vec3(random_double() - 0.5, random_double() - 0.5, 0);
    };

    Ray get_ray(int i, int j) const {
        // TODO: get rid of sample generation
        auto offset = sample_square();
        auto pixel_sample = _top_left_pixel
                          + ((i + offset.get_x()) * _unit_vec_u)
                          + ((j + offset.get_y()) * _unit_vec_v);

        auto ray_origin = _camera;
        auto ray_direction = pixel_sample - ray_origin;

        return Ray(ray_origin, ray_direction);
    }
};

#endif // CAMERA_H_