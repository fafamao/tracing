#ifndef VEC3_POD_CUH_
#define VEC3_POD_CUH_

#include "constants.h"
#include <cmath>
#include <random>

namespace cuda_device {
struct Vec3 {
  float x, y, z;
};

// Unary negation
__device__ __host__ inline Vec3 operator-(const Vec3 &v) {
  return Vec3{-v.x, -v.y, -v.z};
}

// In-place addition
__device__ inline Vec3 &operator+=(Vec3 &u, const Vec3 &v) {
  u.x += v.x;
  u.y += v.y;
  u.z += v.z;
  return u;
}

// In-place multiplication by scalar
__device__ inline Vec3 &operator*=(Vec3 &v, float t) {
  v.x *= t;
  v.y *= t;
  v.z *= t;
  return v;
}

// In-place division by scalar
__device__ inline Vec3 &operator/=(Vec3 &v, float t) { return v *= 1 / t; }

// Vector length squared
__device__ __host__ inline float length_squared(const Vec3 &v) {
  return v.x * v.x + v.y * v.y + v.z * v.z;
}

// Vector length
__device__ __host__ inline float length(const Vec3 &v) {
  return sqrtf(length_squared(v));
}

// Check if vector is close to zero
__device__ inline bool near_zero(const Vec3 &v) {
  auto s = 1e-8f;
  return (fabsf(v.x) < s) && (fabsf(v.y) < s) && (fabsf(v.z) < s);
}

// Vector addition, add host specifier for aabb construction in host
__device__ __host__ inline Vec3 operator+(const Vec3 &u, const Vec3 &v) {
  return Vec3{u.x + v.x, u.y + v.y, u.z + v.z};
}

// Vector subtraction, add host specifier for aabb construction in host
__device__ __host__ inline Vec3 operator-(const Vec3 &u, const Vec3 &v) {
  return Vec3{u.x - v.x, u.y - v.y, u.z - v.z};
}

// Element-wise vector multiplication
__device__ inline Vec3 operator*(const Vec3 &u, const Vec3 &v) {
  return Vec3{u.x * v.x, u.y * v.y, u.z * v.z};
}

// Scalar multiplication (t * v)
__device__ __host__ inline Vec3 operator*(float t, const Vec3 &v) {
  return Vec3{t * v.x, t * v.y, t * v.z};
}

// Scalar multiplication (v * t)
__device__ __host__ inline Vec3 operator*(const Vec3 &v, float t) {
  return t * v;
}

// Scalar division
__device__ __host__ inline Vec3 operator/(const Vec3 &v, float t) {
  return (1 / t) * v;
}

// Dot product
__device__ __host__ inline float dot(const Vec3 &u, const Vec3 &v) {
  return u.x * v.x + u.y * v.y + u.z * v.z;
}

// Cross product
__device__ __host__ inline Vec3 cross(const Vec3 &u, const Vec3 &v) {
  return Vec3{u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z,
              u.x * v.y - u.y * v.x};
}

// Generate a random vector
__device__ inline Vec3 random_vec() {
  return Vec3{random_float(), random_float(), random_float()};
}

// Generate a random vector in a given range
__device__ inline Vec3 random_vec(float min, float max) {
  return Vec3{random_float(min, max), random_float(min, max),
              random_float(min, max)};
}

// Generate a random unit vector using rejection method
__device__ inline Vec3 random_unit_vector() {
  while (true) {
    auto p = random_vec(-1, 1);
    auto len_sq = length_squared(p);
    if (len_sq < 1) {
      return p / length(p);
    }
  }
}

// Generate a random vector on a hemisphere
__device__ inline Vec3 random_on_hemisphere(const Vec3 &normal) {
  Vec3 on_unit_sphere = random_unit_vector();
  // In the same hemisphere as the normal
  if (dot(on_unit_sphere, normal) > 0.0f)
    return on_unit_sphere;
  else
    return -on_unit_sphere;
}

// Get a unit vector
__device__ __host__ inline Vec3 unit_vector(const Vec3 &v) {
  return v / length(v);
}

// Reflect a vector
__device__ inline Vec3 reflect(const Vec3 &v, const Vec3 &n) {
  return v - 2 * dot(v, n) * n;
}

// Refract a vector
__device__ inline Vec3 refract(const Vec3 &uv, const Vec3 &n,
                               float etai_over_etat) {
  auto cos_theta = fminf(dot(-uv, n), 1.0f);
  Vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
  Vec3 r_out_parallel = -sqrtf(fabsf(1.0f - length_squared(r_out_perp))) * n;
  return r_out_perp + r_out_parallel;
}
} // namespace cuda_device
#endif // VEC3_POD_CUH_