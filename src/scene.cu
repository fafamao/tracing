#include "scene.h"
#include "sphere.h"
#include "material.h"
#include "bvh_node.h"

__global__ hittable_list* generate_scene_device()
{
    {
        int n = 500;
        hittable **list = new hittable *[n + 1];
        list[0] = new sphere(Vec3(0, -1000, 0), 1000, new Lambertian(Color(0.5, 0.5, 0.5)));
        int i = 1;
        for (int a = -11; a < 11; a++)
        {
            for (int b = -11; b < 11; b++)
            {
                float choose_mat = drand48();
                Vec3 center(a + 0.9 * drand48(), 0.2, b + 0.9 * drand48());
                if ((center - Vec3(4, 0.2, 0)).get_length() > 0.9)
                {
                    if (choose_mat < 0.8)
                    { // diffuse
                        list[i++] = new sphere(center, 0.2, new Lambertian(Color(drand48() * drand48(), drand48() * drand48(), drand48() * drand48())));
                    }
                    else if (choose_mat < 0.95)
                    { // metal
                        list[i++] = new sphere(center, 0.2,
                                               new Metal(Color(0.5 * (1 + drand48()), 0.5 * (1 + drand48()), 0.5 * (1 + drand48())), 0.5 * drand48()));
                    }
                    else
                    { // glass
                        list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                    }
                }
            }
        }

        list[i++] = new sphere(Vec3(0, 1, 0), 1.0, new dielectric(1.5));
        list[i++] = new sphere(Vec3(-4, 1, 0), 1.0, new Lambertian(Color(0.4, 0.2, 0.1)));
        list[i++] = new sphere(Vec3(4, 1, 0), 1.0, new Metal(Color(0.7, 0.6, 0.5), 0.0));

        return new hittable_list(list, i);
    }
}

void generate_scene_host(hittable_list &world)
{
    auto ground_material = make_shared<Lambertian>(Color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(Vec3(0, -1000, 0), 1000, ground_material));

    for (int a = -11; a < 11; a++)
    {
        for (int b = -11; b < 11; b++)
        {
            auto choose_mat = random_double();
            Vec3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

            if ((center - Vec3(4, 0.2, 0)).get_length() > 0.9)
            {
                shared_ptr<Material> sphere_material;

                if (choose_mat < 0.8)
                {
                    // diffuse
                    Color random_color = Color::random_color();
                    random_color *= Color::random_color();
                    sphere_material = make_shared<Lambertian>(random_color);
                    auto center2 = center + Vec3(0, random_double(0, .5), 0);
                    world.add(make_shared<sphere>(center, center2, 0.2, sphere_material));
                }
                else if (choose_mat < 0.95)
                {
                    // metal
                    auto albedo = Color::random_color(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<Metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
                else
                {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(Vec3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<Lambertian>(Color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(Vec3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<Metal>(Color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(Vec3(4, 1, 0), 1.0, material3));

    world = hittable_list(make_shared<bvh_node>(world));
}