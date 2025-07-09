#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class Ray
{
private:
  Vec3 origin;
  Vec3 dir;
  float time_;

public:
  __host__ __device__ Ray() {};
  __host__ __device__ Ray(const Vec3 &ori, const Vec3 &di, float time) : origin(ori), dir(di), time_(time) {};
  __host__ __device__ Ray(const Vec3 &ori, const Vec3 &di) : origin(ori), dir(di), time_(0.0f) {};
  __host__ __device__ const Vec3 &get_origin() const { return origin; }
  __host__ __device__ const Vec3 &get_direction() const { return dir; }

  // Calculate ray position with given time t
  __host__ __device__ Vec3 at(float t) const
  {
    return origin + t * dir;
  }
  __host__ __device__ float get_time() const
  {
    return time_;
  }
};

#endif
