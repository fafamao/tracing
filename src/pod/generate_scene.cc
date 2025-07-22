#include "generate_scene.h"
#include "constants.h"
#include "sphere_pod.cuh"
#include <utility>

namespace cuda_device
{

  inline Material create_lambertian_material(const Color &albedo)
  {
    Material mat = {};
    mat.type = LAMBERTIAN;
    mat.albedo = albedo;
    return mat;
  }

  inline Material create_metal_material(const Color &albedo, float fuzz)
  {
    Material mat = {};
    mat.type = METAL;
    mat.albedo = albedo;
    mat.metal.fuzz = fuzz;
    return mat;
  }

  inline Material create_dielectric_material(float index_of_refraction)
  {
    Material mat = {};
    mat.type = DIELECTRIC;
    mat.albedo = Color{1.0f, 1.0f, 1.0f};
    mat.dielectric.ir = index_of_refraction;
    return mat;
  }

  std::vector<Hittable> generate_world()
  {
    std::cout << "Generating scene on the host..." << std::endl;
    std::vector<Hittable> world_objects;

    // --- Ground Sphere ---
    Material ground_material =
        create_lambertian_material(Color{0.5f, 0.5f, 0.5f});

    Sphere ground_sphere = create_static_sphere(Vec3{0.0f, -1000.0f, 0.0f},
                                                1000.0f, ground_material);

    world_objects.push_back(create_hittable_from_sphere(&ground_sphere));

    // --- Random Small Spheres ---
    const int spheres_per_axis = 11;
    for (int a = -spheres_per_axis; a < spheres_per_axis; a++)
    {
      for (int b = -spheres_per_axis; b < spheres_per_axis; b++)
      {
        float choose_mat = random_float();
        Vec3 center{a + 0.9f * random_float(), 0.2f, b + 0.9f * random_float()};

        if (length(center - Vec3{4.0f, 0.2f, 0.0f}) > 0.9f)
        {
          Material sphere_material;

          if (choose_mat < 0.8f)
          { // Diffuse
            Color albedo = random_vec() * random_vec();
            sphere_material = create_lambertian_material(albedo);
          }
          else if (choose_mat < 0.95f)
          { // Metal
            Color albedo = random_vec(0.5f, 1.0f);
            float fuzz = random_float() * 0.5f;
            sphere_material = create_metal_material(albedo, fuzz);
          }
          else
          { // Glass
            sphere_material = create_dielectric_material(1.5f);
          }

          Sphere new_sphere = create_static_sphere(center, 0.2f, sphere_material);
          world_objects.push_back(create_hittable_from_sphere(&new_sphere));
        }
      }
    }

    // --- Three Large Spheres ---
    Material material1 = create_dielectric_material(1.5f);
    Sphere sphere1 =
        create_static_sphere(Vec3{0.0f, 1.0f, 0.0f}, 1.0f, material1);
    world_objects.push_back(create_hittable_from_sphere(&sphere1));

    Material material2 = create_lambertian_material(Color{0.4f, 0.2f, 0.1f});
    Sphere sphere2 =
        create_static_sphere(Vec3{-4.0f, 1.0f, 0.0f}, 1.0f, material2);
    world_objects.push_back(create_hittable_from_sphere(&sphere2));

    Material material3 = create_metal_material(Color{0.7f, 0.6f, 0.5f}, 0.0f);
    Sphere sphere3 =
        create_static_sphere(Vec3{4.0f, 1.0f, 0.0f}, 1.0f, material3);
    world_objects.push_back(create_hittable_from_sphere(&sphere3));

    std::cout << "Generated a total of " << world_objects.size() << " objects."
              << std::endl;
    return world_objects;
  }
} // namespace cuda_device