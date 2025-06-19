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
        size_t world_size = 22 * 22 + 1 + 3;
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
                        d_list[i++] = new sphere(center, 0.2, new Dielectric(1.5));
                    }
                }
            }
        }

        d_list[i++] = new sphere(Vec3(0, 1, 0), 1.0, new Dielectric(1.5));
        d_list[i++] = new sphere(Vec3(-4, 1, 0), 1.0, new Lambertian(Color(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(Vec3(4, 1, 0), 1.0, new Metal(Color(0.7, 0.6, 0.5), 0.0));

        hittable **bvh_world;
        *bvh_world = new bvh_node(d_list, world_size, local_rand_state);
        *d_world = new hittable_list(bvh_world, 1);

        *rand_state = local_rand_state;
    }
}

void generate_scene_host(hittable_list &world)
{
    auto ground_material = new Lambertian(Color(0.5, 0.5, 0.5));
    auto object_ground = new sphere(Vec3(0, -1000, 0), 1000, ground_material);
    world.add(object_ground);

    for (int a = -11; a < 11; a++)
    {
        for (int b = -11; b < 11; b++)
        {
            auto choose_mat = random_double();
            Vec3 center(a + 0.9f * random_double(), 0.2f, b + 0.9f * random_double());

            if ((center - Vec3(4, 0.2, 0)).get_length() > 0.9f)
            {
                if (choose_mat < 0.8f)
                {
                    // diffuse
                    Color random_color = Color::random_color();
                    random_color *= Color::random_color();
                    auto center2 = center + Vec3(0, random_double(0, .5), 0);
                    auto sphere_material = new Lambertian(random_color);
                    auto sphere_object = new sphere(center, center2, 0.2, sphere_material);
                    world.add(sphere_object);
                }
                else if (choose_mat < 0.95f)
                {
                    // metal
                    auto albedo = Color::random_color(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    auto sphere_material = new Metal(albedo, fuzz);
                    auto sphere_object = new sphere(center, 0.2, sphere_material);
                    world.add(sphere_object);
                }
                else
                {
                    // glass
                    auto sphere_material = new Dielectric(1.5);
                    auto sphere_object = new sphere(center, 0.2, sphere_material);
                    world.add(sphere_object);
                }
            }
        }
    }

    auto material1 = new Dielectric(1.5);
    auto object1 = new sphere(Vec3(0, 1, 0), 1.0, material1);
    world.add(object1);

    auto material2 = new Lambertian(Color(0.4, 0.2, 0.1));
    auto object2 = new sphere(Vec3(-4, 1, 0), 1.0, material2);
    world.add(object2);

    auto material3 = new Metal(Color(0.7, 0.6, 0.5), 0.0);
    auto object = new sphere(Vec3(4, 1, 0), 1.0, material3);
    world.add(object);

    // Construct BVH nodes
    auto bvh_world = new bvh_node(world);
    world = hittable_list(bvh_world);
}