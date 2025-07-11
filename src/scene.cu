#include "scene.h"

#define RND (curand_uniform(&local_rand_state))

__global__ void generate_scene_device(hittable **d_list, hittable **d_world, curandState *rand_state, Camera **camera, int *d_object_count)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curandState local_rand_state = *rand_state;
        Material *mat = new Lambertian(Color(0.5f, 0.5f, 0.5f));
        d_list[0] = new Sphere(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f, mat);
        int i = 1;
        for (int a = -11; a < 11; a++)
        {
            for (int b = -11; b < 11; b++)
            {
                float choose_mat = RND;
                Vec3 center(a + 0.9f * RND, 0.2f, b + 0.9f * RND);
                if ((center - Vec3(4.0f, 0.2f, 0.0f)).get_length() > 0.9f)
                {
                    Material *mat;
                    if (choose_mat < 0.8f)
                    { // diffuse
                        mat = new Lambertian(Color(RND * RND, RND * RND, RND * RND));
                        d_list[i++] = new Sphere(center, 0.2f, mat);
                    }
                    else if (choose_mat < 0.95f)
                    { // metal
                        mat = new Metal(Color(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND);
                        d_list[i++] = new Sphere(center, 0.2f, mat);
                    }
                    else
                    { // glass
                        mat = new Dielectric(1.5f);
                        d_list[i++] = new Sphere(center, 0.2f, mat);
                    }
                }
            }
        }

        d_list[i++] = new Sphere(Vec3(0, 1, 0), 1.0f, new Dielectric(1.5f));
        d_list[i++] = new Sphere(Vec3(-4, 1, 0), 1.0f, new Lambertian(Color(0.4f, 0.2f, 0.1f)));
        d_list[i++] = new Sphere(Vec3(4, 1, 0), 1.0f, new Metal(Color(0.7f, 0.6f, 0.5f), 0.0f));

        *d_object_count = i;

        // TODO: enable bvh node
        *d_world = new hittable_list(d_list, i - 1);

        *rand_state = local_rand_state;

        // Create camera
        Vec3 camera_origin = Vec3(13, 2, 3);
        Vec3 camera_dest = Vec3(0, 0, 0);
        Vec3 camera_up = Vec3(0, 1, 0);
        *camera = new Camera(camera_origin, camera_dest, camera_up);
    }
}

__global__ void free_scene(hittable **d_list, hittable **d_world, Camera **d_camera, int *d_object_count)
{
    for (int i = 0; i < *d_object_count; i++)
    {
        Material *local_ptr = ((Sphere *)d_list[i])->get_mat_ptr();
        delete local_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

void generate_scene_host(hittable_list &world)
{
    auto ground_material = new Lambertian(Color(0.5f, 0.5f, 0.5f));
    auto object_ground = new Sphere(Vec3(0, -1000, 0), 1000, ground_material);
    world.add(object_ground);

    for (int a = -11; a < 11; a++)
    {
        for (int b = -11; b < 11; b++)
        {
            auto choose_mat = random_float();
            Vec3 center(a + 0.9f * random_float(), 0.2f, b + 0.9f * random_float());

            if ((center - Vec3(4, 0.2f, 0)).get_length() > 0.9f)
            {
                if (choose_mat < 0.8f)
                {
                    // diffuse
                    Color random_color = Color::random_color();
                    random_color *= Color::random_color();
                    auto center2 = center + Vec3(0, random_float(0, .5f), 0);
                    auto sphere_material = new Lambertian(random_color);
                    auto sphere_object = new Sphere(center, center2, 0.2f, sphere_material);
                    world.add(sphere_object);
                }
                else if (choose_mat < 0.95f)
                {
                    // metal
                    auto albedo = Color::random_color(0.5f, 1);
                    auto fuzz = random_float(0, 0.5f);
                    auto sphere_material = new Metal(albedo, fuzz);
                    auto sphere_object = new Sphere(center, 0.2f, sphere_material);
                    world.add(sphere_object);
                }
                else
                {
                    // glass
                    auto sphere_material = new Dielectric(1.5f);
                    auto sphere_object = new Sphere(center, 0.2f, sphere_material);
                    world.add(sphere_object);
                }
            }
        }
    }

    auto material1 = new Dielectric(1.5);
    auto object1 = new Sphere(Vec3(0, 1, 0), 1.0f, material1);
    world.add(object1);

    auto material2 = new Lambertian(Color(0.4f, 0.2f, 0.1f));
    auto object2 = new Sphere(Vec3(-4, 1, 0), 1.0f, material2);
    world.add(object2);

    auto material3 = new Metal(Color(0.7f, 0.6f, 0.5f), 0.0f);
    auto object = new Sphere(Vec3(4, 1, 0), 1.0f, material3);
    world.add(object);

    // Construct BVH nodes
    auto bvh_world = new bvh_node(world);
    world = hittable_list(bvh_world);
}