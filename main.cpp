#include "color.h"
#include "ray.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"

int main()
{
    // Add earth and universe
    hittable_list world;

    auto material_ground = make_shared<Lambertian>(Color(0.8, 0.8, 0.0));
    auto material_center = make_shared<Lambertian>(Color(0.1, 0.2, 0.5));
    auto material_left = make_shared<dielectric>(1.50);
    auto material_bubble = make_shared<dielectric>(1.00 / 1.50);
    auto material_right = make_shared<Metal>(Color(0.8, 0.6, 0.2), 1.0);

    world.add(make_shared<sphere>(Vec3(0.0, -100.5, -1.0), 100.0, material_ground));
    world.add(make_shared<sphere>(Vec3(0.0, 0.0, -1.2), 0.5, material_center));
    world.add(make_shared<sphere>(Vec3(-1.0, 0.0, -1.0), 0.5, material_left));
    world.add(make_shared<sphere>(Vec3(-1.0, 0.0, -1.0), 0.4, material_bubble));
    world.add(make_shared<sphere>(Vec3(1.0, 0.0, -1.0), 0.5, material_right));

    world.add(make_shared<sphere>(Vec3(0.0, -100.5, -1.0), 100.0, material_ground));
    world.add(make_shared<sphere>(Vec3(0.0, 0.0, -1.2), 0.5, material_center));
    world.add(make_shared<sphere>(Vec3(-1.0, 0.0, -1.0), 0.5, material_left));
    world.add(make_shared<sphere>(Vec3(1.0, 0.0, -1.0), 0.5, material_right));
    world.add(make_shared<sphere>(Vec3(-1.0, 0.0, -1.0), 0.4, material_bubble));

    Vec3 camera_origin = Vec3(-2, 2, 1);
    Vec3 camera_dest = Vec3(0, 0, -1);
    Vec3 camera_up = Vec3(0, 1, 0);
    Camera camera(camera_origin, camera_dest, camera_up);
    camera.render(world);

    return 0;
}
