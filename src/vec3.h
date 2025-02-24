#ifndef VEC3_H
#define VEC3_H

#include <cmath>

namespace vec3
{
    class Vec3
    {
    private:
        double pos[3];

    public:
        Vec3() {};
        Vec3(double pos1, double pos2, double pos3) : pos{pos1, pos2, pos3} {};
        ~Vec3() {};

        double get_x() const
        {
            return pos[0];
        }
        double get_y() const
        {
            return pos[1];
        }
        double get_z() const
        {
            return pos[2];
        }

        double operator[](int i) const
        {
            return pos[i];
        }
        double &operator[](int i) { return pos[i]; }

        Vec3 &operator+=(const Vec3 &vec)
        {
            pos[0] += vec.get_x();
            pos[1] += vec.get_y();
            pos[2] += vec.get_z();
            return *this;
        }

        Vec3 &operator*=(const Vec3 &vec)
        {
            pos[0] *= vec.get_x();
            pos[1] *= vec.get_y();
            pos[2] *= vec.get_z();
            return *this;
        }

        double get_length() const
        {
            return std::sqrt(length_squared());
        }

        double length_squared() const
        {
            return pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2];
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