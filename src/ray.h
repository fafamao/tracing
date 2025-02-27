#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class Ray
{
private:
  Vec3 origin;
  Vec3 dir;

public:
  Ray() {};
  Ray(const Vec3 &ori, const Vec3 &di) : origin(ori), dir(di) {};
  const Vec3 &get_origin() const { return origin; }
  const Vec3 &get_direction() const { return dir; }

  // Calculate ray position with given time t
  Vec3 at(double t) const
  {
    return origin + t * dir;
  }
};

#endif
