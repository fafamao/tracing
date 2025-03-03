#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include "constants.h"

class Vec3
{
private:
    double x, y, z;

public:
    Vec3() {};
    Vec3(double pos1, double pos2, double pos3) : x{pos1}, y{pos2}, z{pos3} {};
    ~Vec3() {};

    double get_x() const
    {
        return x;
    }
    double get_y() const
    {
        return y;
    }
    double get_z() const
    {
        return z;
    }

    Vec3 operator-() const
    {
        return Vec3(-x, -y, -z);
    }

    Vec3 &operator+=(const Vec3 &vec)
    {
        x += vec.get_x();
        y += vec.get_y();
        z += vec.get_z();
        return *this;
    }

    Vec3 &operator*=(const Vec3 &vec)
    {
        x *= vec.get_x();
        y *= vec.get_y();
        z *= vec.get_z();
        return *this;
    }

    double get_length() const
    {
        return std::sqrt(length_squared());
    }

    double length_squared() const
    {
        return x * x + y * y + z * z;
    }

    static Vec3 random()
    {
        return Vec3(random_double(), random_double(), random_double());
    }
    static Vec3 random(double min, double max)
    {
        return Vec3(random_double(min, max), random_double(min, max), random_double(min, max));
    }
};
inline Vec3 operator+(const Vec3 &vec1, const Vec3 &vec2)
{
    return Vec3(vec1.get_x() + vec2.get_x(), vec1.get_y() + vec2.get_y(), vec1.get_z() + vec2.get_z());
}
inline Vec3 operator-(const Vec3 &vec1, const Vec3 &vec2)
{
    return Vec3(vec1.get_x() - vec2.get_x(), vec1.get_y() - vec2.get_y(), vec1.get_z() - vec2.get_z());
}
inline Vec3 operator*(const Vec3 &vec1, const Vec3 &vec2)
{
    return Vec3(vec1.get_x() * vec2.get_x(), vec1.get_y() * vec2.get_y(), vec1.get_z() * vec2.get_z());
}
inline Vec3 operator*(const double scale, const Vec3 &vec2)
{
    return Vec3(scale * vec2.get_x(), scale * vec2.get_y(), scale * vec2.get_z());
}
inline Vec3 operator*(const Vec3 &vec, const double scale)
{
    return scale * vec;
}
inline Vec3 operator/(const Vec3 &vec, double scale)
{
    return (1 / scale) * vec;
}
inline double dot(const Vec3 &vec1, const Vec3 &vec2)
{
    return vec1.get_x() * vec2.get_x() + vec1.get_y() * vec2.get_y() + vec1.get_z() * vec2.get_z();
}
inline Vec3 unit_vector(const Vec3 &v)
{
    return v / v.get_length();
}

// Generate random unit vector
inline Vec3 random_unit_vec_rejection_method()
{
    while (true)
    {
        auto p = Vec3::random(-1, 1);
        auto lensq = p.length_squared();
        if (1e-160 < lensq && lensq <= 1)
            return p / sqrt(lensq);
    }
}

inline Vec3 random_unit_vec_spherical_coordinates()
{
    double theta = random_double(0, 2 * PI);
    double phi = acos(2 * random_double() - 1);
    double x = sin(phi) * cos(theta);
    double y = sin(phi) * sin(theta);
    double z = cos(phi);
    return Vec3(x, y, z);
}

inline Vec3 random_unit_vec_normal_distribution()
{
    while (true)
    {
        std::random_device rd;
        std::mt19937 rng(rd());
        std::normal_distribution<double> dist(0.0, 1.0);
        double x = dist(rng);
        double y = dist(rng);
        double z = dist(rng);
        auto vec = Vec3(x, y, z);
        auto len = vec.length_squared();
        if (1e-160 < len && len <= 1)
        {
            return vec / sqrt(len);
        }
    }
}

inline Vec3 random_on_hemisphere(const Vec3 &normal)
{
    Vec3 on_unit_sphere = random_unit_vec_rejection_method();
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

#endif // VEC3_H