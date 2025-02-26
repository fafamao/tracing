#ifndef VEC3_H
#define VEC3_H

#include <cmath>

namespace vec3
{
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

        Vec3 operator-() const {
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
    inline Vec3 unit_vector(const Vec3& v) {
        return v / v.get_length();
    }
}

#endif // VEC3_H