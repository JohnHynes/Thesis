#ifndef GPU_RAY_TRACER_HITTABLE_HPP_
#define GPU_RAY_TRACER_HITTABLE_HPP_

#include "Preprocessor.hpp"
#include "Ray.hpp"
#include "constants.hpp"
#include "types.hpp"
#include <glm/geometric.hpp>
#include <limits>

class material;

struct hit_record
{
  point3 point;
  vec3 normal;
  int mat_idx;
  num t;
  bool front_face;

  __host__ __device__
  inline void
  set_face_normal (const ray &r, const vec3 &outward_normal)
  {
    front_face = glm::dot (r.dir, outward_normal) < CONST(0.0);
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
  Plane,
  Rectangle,
  Triangle,
  Unknown
};

class hittable
{
protected:
  __host__ __device__
  hittable (object_id i) : id (i)
  {
  }
public:

  __host__ __device__
  virtual ~hittable() {
  }

  __host__ __device__
  virtual bool
  hit (const ray &r, num tmin, num tmax, hit_record &hitrec) const = 0;

  // Forward declaration of size_of
  __host__ __device__
  static int
  size_of (hittable *m);

  // Forward declaration of make_from
  __host__ __device__
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
  __host__ __device__
  sphere (const vec3 &c, num r, int m)
    : hittable(object_id::Sphere), center (c), radius (r), mat_idx (m)
  {
  }

  __host__ __device__
  virtual ~sphere() override {
  }

  // Member Functions
  __host__ __device__
  virtual bool
  hit (const ray &r, num tmin, num tmax, hit_record &hitrec) const override
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
//                                Plane                                  //
///////////////////////////////////////////////////////////////////////////

class plane : public hittable
{
public:
  point3 p;
  vec3 surface_normal;
  int mat_idx;
public:
  __host__ __device__
  plane(point3 _p, vec3 _sn, int m)
    : hittable(object_id::Plane), p(_p), surface_normal(-glm::normalize(_sn)), mat_idx(m) {
  }

  __host__ __device__
  virtual ~plane() override {
  }

  __host__ __device__
  virtual bool
  hit(const ray& r, num t_min, num t_max, hit_record& rec) const override
  {
    num numerator = glm::dot(p - r.orig, surface_normal);
    num t = numerator / glm::dot(r.dir, surface_normal);
    if (t < t_min || t_max < t)
    {
      return false;
    }
    rec.t = t;
    rec.point = r.at(t);
    rec.mat_idx = mat_idx;
    rec.set_face_normal(r, surface_normal);
    return true;
  }
};


///////////////////////////////////////////////////////////////////////////
//                              Rectangle                                //
///////////////////////////////////////////////////////////////////////////

class rectangle : public hittable
{
public:
  int mat_idx;
  point3 p1, p2, p3;
  vec3 surface_normal;

public:
  __host__ __device__
  rectangle(point3 _p1, point3 _p2, point3 _p3, int m)
    : hittable(object_id::Rectangle)
    , mat_idx(m)
    , p1(_p1)
    , p2(_p2)
    , p3(_p3)
    , surface_normal(glm::normalize(glm::cross(p1 - p2, p3 - p2)))
  {
  }


  __host__ __device__
  virtual ~rectangle() override {
  }

  __host__ __device__
  virtual bool
  hit(const ray& r, num t_min, num t_max, hit_record& rec) const override
  {
    // TODO : Implement hit function.
    return false;
  }
};

///////////////////////////////////////////////////////////////////////////
//                              Triangle                                 //
///////////////////////////////////////////////////////////////////////////

class triangle : public hittable
{
public:
  int mat_idx;
  point3 p1, p2, p3;
  vec3 surface_normal;
public:
  __host__ __device__
  triangle(point3 _p1, point3 _p2, point3 _p3, int m)
    : hittable(object_id::Triangle)
    , mat_idx(m)
    , p1(_p1)
    , p2(_p2)
    , p3(_p3)
    , surface_normal (glm::normalize(glm::cross(p1 - p2, p3 - p2))) {
  }

  __host__ __device__
  virtual ~triangle() override {
  }

  __host__ __device__
  virtual bool
  hit(const ray& r, num t_min, num t_max, hit_record& rec) const override
  {
    // TODO : Implement hit function.
    // 1. Use the Moller-Trumbore ray-triangle algorithm to compute t
    num epsilon = CONST(0.0000001);
    vec3 edge1 = p2 - p1;
    vec3 edge2 = p3 - p1;
    vec3 h = glm::cross(r.dir, edge2);
    num a = glm::dot(h, edge1);
    if (-epsilon < a && a < epsilon) // Ray is approximately parallel
    {
      return false;
    }
    num f = CONST(1.0) / a;
    vec3 s = r.orig - p1;
    num u = f * glm::dot(h, s);
    if (u < 0 || 1 < u)
    {
      return false;
    }
    vec3 q = glm::cross(s, edge1);
    num v = f * glm::dot(r.dir, q);
    if (v < CONST(0) || CONST(1) < u + v)
    {
      return false;
    }
    num t = f * glm::dot(edge2, q);
    if (t < t_min || t_max < t)
    {
      return false;
    }
    // 2. Assign proper values to hitrec
    rec.t = t;
    rec.point = r.at(t);
    rec.mat_idx = mat_idx;
    rec.set_face_normal(r, surface_normal);
    return true;
  }
};

///////////////////////////////////////////////////////////////////////////
//                          Hittable Functions                           //
///////////////////////////////////////////////////////////////////////////

__host__ __device__
inline
int
hittable::size_of (hittable *o)
{
  switch (o->id)
  {
    case object_id::Sphere:
      return sizeof (sphere);
    case object_id::Plane:
      return sizeof (plane);
    case object_id::Triangle:
      return sizeof (triangle);
    default:
      return 0;
  }
}

__host__ __device__
inline
hittable *
hittable::make_from (hittable *old)
{
  switch (old->id)
  {
    case object_id::Sphere:
      return new sphere (*reinterpret_cast<sphere *> (old));
    case object_id::Plane:
      return new plane (*reinterpret_cast<plane *> (old));
    case object_id::Triangle:
      return new triangle (*reinterpret_cast<triangle *> (old));
    default:
      return nullptr;
  }
}

#endif