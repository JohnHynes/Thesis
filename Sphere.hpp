#pragma once

#include "Hittable.hpp"
#include "Ray.hpp"

class sphere : public hittable
{
public:
  vec3 center;
  num radius;
  std::shared_ptr<material> mat_ptr;

public:
  // Constructors
  sphere () = default;
  sphere (sphere const &) = default;
  sphere (sphere &&) = default;
  sphere &
  operator= (sphere const &) = default;
  sphere &
  operator= (sphere &&) = default;

  sphere (const vec3 &c, num r, std::shared_ptr<material> m)
    : center (c), radius (r), mat_ptr (m)
  {
  }

  // Member Functions
  bool
  hit (const ray &r, num tmin, num tmax, hit_record &hitrec) const
  {
    vec3 oc = r.origin() - center;
    num a = glm::dot (r.dir, r.dir);
    num half_b = glm::dot (oc, r.dir);
    num c = glm::dot (oc, oc) - radius * radius;
    num discriminant = half_b * half_b - a * c;

    if (discriminant <= 0)
    {
      return false;
    }

    num root = sqrt (discriminant);

    num t = (-half_b - root) / a;
    if (tmin < t && t < tmax)
    {
      hitrec.t = t;
      hitrec.point = r.at (hitrec.t);
      glm::vec3 outward_normal = (hitrec.point - center) / radius;
      hitrec.set_face_normal (r, outward_normal);
      hitrec.mat_ptr = mat_ptr;
      return true;
    }

    t = (-half_b + root) / a;
    if (tmin < t && t < tmax)
    {
      hitrec.t = t;
      hitrec.point = r.at (hitrec.t);
      glm::vec3 outward_normal = (hitrec.point - center) / radius;
      hitrec.set_face_normal (r, outward_normal);
      hitrec.mat_ptr = mat_ptr;
      return true;
    }
    return false;
  }
};