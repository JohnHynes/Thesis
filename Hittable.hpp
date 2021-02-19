#pragma once

#include "Ray.hpp"
#include "types.hpp"

class material;

struct hit_record
{
  point3 point;
  vec3 normal;
  int mat_idx;
  num t;
  bool front_face;

  HOST_DEVICE
  inline void
  set_face_normal (const ray &r, const vec3 &outward_normal)
  {
    front_face = glm::dot (r.dir, outward_normal) < 0;
    if (front_face)
    {
      normal = outward_normal;
    }
    else
    {
      normal = -outward_normal;
    }
  }
};

enum class object_id
{
  Sphere,
  Unknown
};

class hittable
{
protected:
  HOST_DEVICE
  hittable (object_id i) : id (i)
  {
  }
public:

  HOST_DEVICE
  virtual bool
  hit (const ray &r, num tmin, num tmax, hit_record &hitrec) const = 0;

  // Forward declaration of size_of
  HOST_DEVICE
  static int
  size_of (hittable *m);

  // Forward declaration of make_from
  HOST_DEVICE
  static hittable *
  make_from (hittable *old);

public:
  object_id id;
};

///////////////////////////////////////////////////////////////////////////
//                                Sphere                                 //
///////////////////////////////////////////////////////////////////////////
class sphere : public hittable
{
public:
  vec3 center;
  num radius;
  int mat_idx;

public:
  // Constructors
  HOST_DEVICE
  sphere (const vec3 &c, num r, int m)
    : hittable(object_id::Sphere), center (c), radius (r), mat_idx (m)
  {
  }

  // Member Functions
  HOST_DEVICE
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
      hitrec.mat_idx = mat_idx;
      return true;
    }

    t = (-half_b + root) / a;
    if (tmin < t && t < tmax)
    {
      hitrec.t = t;
      hitrec.point = r.at (hitrec.t);
      glm::vec3 outward_normal = (hitrec.point - center) / radius;
      hitrec.set_face_normal (r, outward_normal);
      hitrec.mat_idx = mat_idx;
      return true;
    }
    return false;
  }
};

///////////////////////////////////////////////////////////////////////////
//                          Hittable Functions                           //
///////////////////////////////////////////////////////////////////////////

HOST
int
hittable::size_of (hittable *o)
{
  switch (o->id)
  {
    case object_id::Sphere:
      return sizeof (sphere);
    default:
      return 0;
  }
}

HOST_DEVICE
hittable *
hittable::make_from (hittable *old)
{
  switch (old->id)
  {
    case object_id::Sphere:
      return new sphere (*reinterpret_cast<sphere *> (old));
    default:
      return nullptr;
  }
}