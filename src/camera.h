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
    void initialize();
    Color ray_color(const Ray &r, const hittable_list &world);

public:
    Camera()
    {
        initialize();
    }
    void render(const hittable_list &world);
};

#endif // CAMERA_H_