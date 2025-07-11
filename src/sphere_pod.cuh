#ifndef SPHERE_POD_CUH_
#define SPHERE_POD_CUH_

#include "vec3_pod.cuh"
#include "material_pod.cuh"
#include "aabb_pod.cuh"

struct Sphere
{
    Vec3 center0; // Starting center for motion
    float radius;
    Material mat; // Material data is embedded directly

    // Data for moving spheres
    bool is_moving;
    Vec3 center_vec; // The vector of movement (center1 - center0)
};

__device__ inline Sphere create_static_sphere(const Vec3 &center, float radius, const Material &mat)
{
    Sphere s;
    s.center0 = center;
    s.radius = fmaxf(0.0f, radius);
    s.mat = mat;
    s.is_moving = false;
    s.center_vec = Vec3{0, 0, 0};
    return s;
}

// Creates a moving sphere
__device__ inline Sphere create_moving_sphere(const Vec3 &center0, const Vec3 &center1, float radius, const Material &mat)
{
    Sphere s;
    s.center0 = center0;
    s.radius = fmaxf(0.0f, radius);
    s.mat = mat;
    s.is_moving = true;
    s.center_vec = center1 - center0;
    return s;
}

// Gets the center of the sphere at a specific time for motion blur
__device__ inline Vec3 sphere_center(const Sphere &s, float time)
{
    if (!s.is_moving)
    {
        return s.center0;
    }
    return s.center0 + time * s.center_vec;
}

// Calculates the bounding box for a sphere
__device__ inline aabb bounding_box_sphere(const Sphere &s)
{
    Vec3 rvec = Vec3{s.radius, s.radius, s.radius};
    if (!s.is_moving)
    {
        return create_aabb_from_points(s.center0 - rvec, s.center0 + rvec);
    }
    // For moving spheres, enclose the boxes at time0 and time1
    aabb box0 = create_aabb_from_points(s.center0 - rvec, s.center0 + rvec);
    aabb box1 = create_aabb_from_points((s.center0 + s.center_vec) - rvec, (s.center0 + s.center_vec) + rvec);
    return create_aabb_from_boxes(box0, box1);
}

// The main ray-sphere intersection logic
__device__ inline bool hit_sphere(
    const Sphere &s, const Ray &r, Interval ray_t, HitRecord &rec)
{
    Vec3 center_at_time = sphere_center(s, r.time);
    Vec3 oc = r.origin - center_at_time;

    auto a = length_squared(r.direction);
    auto h = dot(r.direction, -oc); // Changed oc to -oc to simplify discriminant calc
    auto c = length_squared(oc) - s.radius * s.radius;

    auto discriminant = h * h - a * c;
    if (discriminant < 0)
    {
        return false;
    }

    auto sqrtd = sqrtf(discriminant);

    // Find the nearest root that lies in the acceptable range
    auto root = (h - sqrtd) / a;
    if (!interval_surrounds(ray_t, root))
    {
        root = (h + sqrtd) / a;
        if (!interval_surrounds(ray_t, root))
        {
            return false;
        }
    }

    // A valid hit was found, so populate the HitRecord
    rec.t = root;
    rec.p = ray_at(r, rec.t);
    Vec3 outward_normal = (rec.p - center_at_time) / s.radius;
    set_face_normal(rec, r, outward_normal);
    rec.mat = s.mat; // Embed the material data directly

    return true;
}

#endif // SPHERE_POD_CUH_