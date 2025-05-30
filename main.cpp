#include "color.h"
#include "ray.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"
#include "mem_pool.h"
#include "bvh_node.h"
#include <cuda_runtime.h>
#include <cstring>

bool is_gpu_available() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if(error_id != cudaSuccess) {
        printf("GPU card is not available.");
        return false;
    }
    return deviceCount > 0;
}

int main()
{
    // Instantiate memory pool
    // TODO: fix this
    size_t rgb_size = PIXEL_HEIGHT * PIXEL_WIDTH * 3;
    size_t pool_siz = rgb_size * 2;
    MemoryPool mem_pool(pool_siz);
    char *pixel_buffer = mem_pool.allocate(rgb_size);

    // Initialize pixel_buffer all white
    memset(pixel_buffer, 255, rgb_size);

/*     // --- Construct the ffplay command ---
    std::string cmd = "ffplay ";
    cmd += "-v error "; // Quieter log level (shows errors only)
    // Input options - must match the data being piped
    cmd += "-f rawvideo ";
    cmd += "-pixel_format rgb24 "; // TODO: define format in constants.h
    cmd += "-video_size " + std::to_string(PIXEL_WIDTH) + "x" + std::to_string(PIXEL_HEIGHT) + " ";
    cmd += "-framerate " + std::to_string(FRAME_RATE) + " ";
    // Other options
    cmd += "-window_title \"Ray Tracer Output (ffplay)\" "; // Optional: Set window title
    cmd += "-i - ";                                         // Read input from stdin

    std::cout << "Executing command: " << cmd << std::endl;

    FILE *pipe = nullptr;

    pipe = popen(cmd.c_str(), "w");
    if (!pipe)
    {
        throw std::runtime_error("popen() to ffplay failed! Is ffplay installed and in PATH?");
    }
    std::cout << "ffplay process started. Rendering frames..." << std::endl;

    if (pipe)
    {
        pclose(pipe);
    } */

    // Instantiate thread pool
    ThreadPool tp;

    // Add earth and universe
    hittable_list world;

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

    Vec3 camera_origin = Vec3(13, 2, 3);
    Vec3 camera_dest = Vec3(0, 0, 0);
    Vec3 camera_up = Vec3(0, 1, 0);
    Camera camera(camera_origin, camera_dest, camera_up, &tp);
    camera.render(world, pixel_buffer);

    printf("Rendering ends\n");

    return 0;
}
