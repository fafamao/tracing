#include "color.h"
#include "ray.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"

int main()
{
    // Add earth and universe
    hittable_list world;
    world.add(make_shared<sphere>(Vec3(0, 0, -1), 0.5));
    world.add(make_shared<sphere>(Vec3(0, -100.5, -1), 100));

    Camera camera;
    camera.render(world);

    return 0;
}
