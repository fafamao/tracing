#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <stdexcept>
#include "constants.h"

class Vec3
{
private:
    double x[3];

public:
    __host__ __device__ Vec3() {};
    __host__ __device__ Vec3(double pos1, double pos2, double pos3) : x{pos1, pos2, pos3} {};
    __host__ __device__ ~Vec3(){};

    __host__ __device__ double get_x() const
    {
        return x[0];
    }
    __host__ __device__ double get_y() const
    {
        return x[1];
    }
    __host__ __device__ double get_z() const
    {
        return x[2];
    }

    __host__ __device__ double operator[](size_t i) const
    {
        return x[i];
    }

    __host__ __device__ Vec3 operator-() const
    {
        return Vec3(-x[0], -x[1], -x[2]);
    }

    __host__ __device__ Vec3 &operator+=(const Vec3 &vec)
    {
        x[0] += vec.get_x();
        x[1] += vec.get_y();
        x[2] += vec.get_z();
        return *this;
    }

    __host__ __device__ Vec3 &operator*=(const Vec3 &vec)
    {
        x[0] *= vec.get_x();
        x[1] *= vec.get_y();
        x[2] *= vec.get_z();
        return *this;
    }

    __host__ __device__ double get_length() const
    {
        return std::sqrt(length_squared());
    }

    __host__ __device__ double length_squared() const
    {
        return x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
    }

    __host__ __device__ static Vec3 random()
    {
        return Vec3(random_double(), random_double(), random_double());
    }
    __host__ __device__ static Vec3 random(double min, double max)
    {
        return Vec3(random_double(min, max), random_double(min, max), random_double(min, max));
    }
    __host__ __device__ bool near_zero() const
    {
        // Return true if the vector is close to zero in all dimensions.
        auto s = 1e-8;
#ifdef __CUDA_ARCH__
        return (fabs(x[0]) < s) && (fabs(x[1]) < s) && (fabs(x[2]) < s);
#else
        return (std::fabs(x[0]) < s) && (std::fabs(x[1]) < s) && (std::fabs(x[2]) < s);
#endif
    }
};
__host__ __device__ inline Vec3 operator+(const Vec3 &vec1, const Vec3 &vec2)
{
    return Vec3(vec1.get_x() + vec2.get_x(), vec1.get_y() + vec2.get_y(), vec1.get_z() + vec2.get_z());
}
__host__ __device__ inline Vec3 operator-(const Vec3 &vec1, const Vec3 &vec2)
{
    return Vec3(vec1.get_x() - vec2.get_x(), vec1.get_y() - vec2.get_y(), vec1.get_z() - vec2.get_z());
}
__host__ __device__ inline Vec3 operator*(const Vec3 &vec1, const Vec3 &vec2)
{
    return Vec3(vec1.get_x() * vec2.get_x(), vec1.get_y() * vec2.get_y(), vec1.get_z() * vec2.get_z());
}
__host__ __device__ inline Vec3 operator*(const double scale, const Vec3 &vec2)
{
    return Vec3(scale * vec2.get_x(), scale * vec2.get_y(), scale * vec2.get_z());
}
__host__ __device__ inline Vec3 operator*(const Vec3 &vec, const double scale)
{
    return scale * vec;
}
__host__ __device__ inline Vec3 operator/(const Vec3 &vec, double scale)
{
    return (1 / scale) * vec;
}
__host__ __device__ inline double dot(const Vec3 &vec1, const Vec3 &vec2)
{
    return vec1.get_x() * vec2.get_x() + vec1.get_y() * vec2.get_y() + vec1.get_z() * vec2.get_z();
}
__host__ __device__ inline Vec3 unit_vector(const Vec3 &v)
{
    return v / v.get_length();
}
__host__ __device__ inline Vec3 reflect(const Vec3 &v, const Vec3 &n)
{
    return v - 2 * dot(v, n) * n;
}
__host__ __device__ inline Vec3 cross(const Vec3 &u, const Vec3 &v)
{
    return Vec3(u.get_y() * v.get_z() - u.get_z() * v.get_y(),
                u.get_z() * v.get_x() - u.get_x() * v.get_z(),
                u.get_x() * v.get_y() - u.get_y() * v.get_x());
}
// TODO: validate calculation from Snil's law
__host__ __device__ inline Vec3 refract(const Vec3 &uv, const Vec3 &n, double etai_over_etat)
{
    // TODO
    auto cos_theta = fmin(dot(-uv, n), 1.0);
    Vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    Vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

// Generate random unit vector
__host__ __device__ inline Vec3 random_unit_vec_rejection_method()
{
    while (true)
    {
        auto p = Vec3::random(-1, 1);
        auto lensq = p.length_squared();
        if (1e-160 < lensq && lensq <= 1)
            return p / sqrt(lensq);
    }
}

__device__ __host__ inline Vec3 random_unit_vec_spherical_coordinates()
{
    double theta = random_double(0, 2 * PI);
    double phi = acos(2 * random_double() - 1);
    double x[3] = {sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi)};
    return Vec3(x[0], x[1], x[2]);
}

[[maybe_unused]] inline Vec3 random_unit_vec_normal_distribution()
{
    while (true)
    {
        std::random_device rd;
        std::mt19937 rng(rd());
        std::normal_distribution<double> dist(0.0, 1.0);
        double x[3] = {dist(rng), dist(rng), dist(rng)};
        auto vec = Vec3(x[0], x[1], x[2]);
        auto len = vec.length_squared();
        if (1e-160 < len && len <= 1)
        {
            return vec / sqrt(len);
        }
    }
}

__device__ __host__ inline Vec3 random_unit_vec_random_cosine_direction()
{
    double u = random_double();
    double v = random_double();
    double theta = acos(sqrt(1 - u));
    double phi = 2 * M_PI * v;
    double x[3] = {sin(theta) * cos(phi), sin(theta) * sin(phi), sqrt(u)};
    return Vec3(x[0], x[1], x[2]);
}

// TODO: use reflection ray that is close to normal(random_cosine_direction)
__device__ __host__ inline Vec3 random_on_hemisphere(const Vec3 &normal)
{
    Vec3 on_unit_sphere = random_unit_vec_rejection_method();
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

#endif // VEC3_H