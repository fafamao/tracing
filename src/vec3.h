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
            pos[0] += vec.get_y();
            pos[0] += vec.get_z();
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
    Vec3 operator+(const Vec3 &vec1, const Vec3 &vec2)
    {
        return Vec3(vec1.get_x() + vec2.get_x(), vec1.get_y() + vec2.get_y(), vec1.get_z() + vec2.get_z());
    }
}

#endif // VEC3_H