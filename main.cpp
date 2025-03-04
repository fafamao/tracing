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
    auto material_left   = make_shared<Metal>(Color(0.8, 0.8, 0.8));
    auto material_right  = make_shared<Metal>(Color(0.8, 0.6, 0.2));

    world.add(make_shared<sphere>(Vec3( 0.0, -100.5, -1.0), 100.0, material_ground));
    world.add(make_shared<sphere>(Vec3( 0.0,    0.0, -1.2),   0.5, material_center));
    world.add(make_shared<sphere>(Vec3(-1.0,    0.0, -1.0),   0.5, material_left));
    world.add(make_shared<sphere>(Vec3( 1.0,    0.0, -1.0),   0.5, material_right));

    Camera camera;
    camera.render(world);

    return 0;
}
