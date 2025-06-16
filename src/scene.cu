#include "scene.h"
#include "sphere.h"
#include "material.h"
#include "bvh_node.h"
#include "random_number_generator.cuh"

#define RND (curand_uniform(&local_rand_state))

__global__ void generate_scene_device(hittable **d_list, hittable **d_world, curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(Vec3(0, -1000, 0), 1000, new Lambertian(Color(0.5, 0.5, 0.5)));
        int i = 1;
        for (int a = -11; a < 11; a++)
        {
            for (int b = -11; b < 11; b++)
            {
                float choose_mat = RND;
                Vec3 center(a + 0.9f * RND, 0.2, b + 0.9f * RND);
                if ((center - Vec3(4, 0.2, 0)).get_length() > 0.9)
                {
                    if (choose_mat < 0.8f)
                    { // diffuse
                        d_list[i++] = new sphere(center, 0.2, new Lambertian(Color(RND * RND, RND * RND, RND * RND)));
                    }
                    else if (choose_mat < 0.95f)
                    { // metal
                        d_list[i++] = new sphere(center, 0.2,
                                               new Metal(Color(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
                    }
                    else
                    { // glass
                        d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                    }
                }
            }
        }

        d_list[i++] = new sphere(Vec3(0, 1, 0), 1.0, new dielectric(1.5));
        d_list[i++] = new sphere(Vec3(-4, 1, 0), 1.0, new Lambertian(Color(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(Vec3(4, 1, 0), 1.0, new Metal(Color(0.7, 0.6, 0.5), 0.0));

        *d_world  = new hittable_list(d_list, 22*22+1+3);

        // TODO: make world into bvh nodes.

        *rand_state = local_rand_state;
    }
}

void generate_scene_host(hittable_list &world)
{
    auto ground_material = std::make_shared<Lambertian>(Color(0.5, 0.5, 0.5));
    world.add(std::make_shared<sphere>(Vec3(0, -1000, 0), 1000, ground_material));

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
                    sphere_material = std::make_shared<Lambertian>(random_color);
                    auto center2 = center + Vec3(0, random_double(0, .5), 0);
                    world.add(std::make_shared<sphere>(center, center2, 0.2, sphere_material));
                }
                else if (choose_mat < 0.95)
                {
                    // metal
                    auto albedo = Color::random_color(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = std::make_shared<Metal>(albedo, fuzz);
                    world.add(std::make_shared<sphere>(center, 0.2, sphere_material));
                }
                else
                {
                    // glass
                    sphere_material = std::make_shared<dielectric>(1.5);
                    world.add(std::make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = std::make_shared<dielectric>(1.5);
    world.add(std::make_shared<sphere>(Vec3(0, 1, 0), 1.0, material1));

    auto material2 = std::make_shared<Lambertian>(Color(0.4, 0.2, 0.1));
    world.add(std::make_shared<sphere>(Vec3(-4, 1, 0), 1.0, material2));

    auto material3 = std::make_shared<Metal>(Color(0.7, 0.6, 0.5), 0.0);
    world.add(std::make_shared<sphere>(Vec3(4, 1, 0), 1.0, material3));

    world = hittable_list(std::make_shared<bvh_node>(world));
}