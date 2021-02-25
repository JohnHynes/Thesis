#ifndef GPU_RAY_TRACER_HITTABLE_HPP_
#define GPU_RAY_TRACER_HITTABLE_HPP_

#include "Preprocessor.hpp"
#include "Ray.hpp"
#include "constants.hpp"
#include "types.hpp"
#include <glm/geometric.hpp>
#include <limits>

class hittable;

struct hit_record
{
  point3 point;
  vec3 normal;
  int mat_idx;
  num t;
  bool front_face;

  __host__ __device__ inline void
  set_face_normal (const ray& r, const vec3& outward_normal)
  {
    front_face = glm::dot (r.dir, outward_normal) < CONST (0.0);
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

enum class hittable_id
{
  Sphere,
  Plane,
  Rectangle,
  Triangle,
  Unknown
};

struct IHittable
{

  __host__ __device__ virtual bool
  hit (const ray& r, num tmin, num tmax, hit_record& hitrec) const = 0;
};

struct sphere : public IHittable
{
  __host__ __device__ inline sphere (const vec3& c, num r, int m)
    : center (c), radius (r), mat_idx (m)
  {
  }

  // Member Functions
  __host__ __device__ inline bool
  hit (const ray& r, num tmin, num tmax, hit_record& hitrec) const override
  {
    vec3 oc = r.origin () - center;
    num a = glm::dot (r.dir, r.dir);
    num half_b = glm::dot (oc, r.dir);
    num c = glm::dot (oc, oc) - radius * radius;
    num discriminant = half_b * half_b - a * c;

    if (discriminant <= 0)
    {
      return false;
    }

    num root = sqrtf (discriminant);

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

  vec3 center;
  num radius;
  int mat_idx;
};

struct plane : public IHittable
{

  __host__ __device__
  plane (point3 p, vec3 sn, int m)
    : p (p), surface_normal (-glm::normalize (sn)), mat_idx (m)
  {
  }

  __host__ __device__ inline bool
  hit (const ray& r, num t_min, num t_max, hit_record& rec) const override
  {
    num numerator = glm::dot (p - r.orig, surface_normal);
    num t = numerator / glm::dot (r.dir, surface_normal);
    if (t < t_min || t_max < t)
    {
      return false;
    }
    rec.t = t;
    rec.point = r.at (t);
    rec.mat_idx = mat_idx;
    rec.set_face_normal (r, surface_normal);
    return true;
  }

  point3 p;
  vec3 surface_normal;
  int mat_idx;
};

struct rectangle : public IHittable
{

  __host__ __device__
  rectangle (point3 _p1, point3 _p2, point3 _p3, int m)
    : mat_idx (m)
    , p1 (_p1)
    , p2 (_p2)
    , p3 (_p3)
    , surface_normal (glm::normalize (glm::cross (p1 - p2, p3 - p2)))
  {
  }

  __host__ __device__ inline bool
  hit (const ray& r, num t_min, num t_max, hit_record& rec) const override
  {
    // TODO : Implement hit function.
    return false;
  }

public:
  int mat_idx;
  point3 p1, p2, p3;
  vec3 surface_normal;
};

struct triangle : public IHittable
{

  __host__ __device__
  triangle (point3 _p1, point3 _p2, point3 _p3, int m)
    : mat_idx (m)
    , p1 (_p1)
    , p2 (_p2)
    , p3 (_p3)
    , surface_normal (glm::normalize (glm::cross (p1 - p2, p3 - p2)))
  {
  }

  __host__ __device__ virtual bool
  hit (const ray& r, num t_min, num t_max, hit_record& rec) const override
  {
    // TODO : Implement hit function.
    // 1. Use the Moller-Trumbore ray-triangle algorithm to compute t
    num epsilon = CONST (0.0000001);
    vec3 edge1 = p2 - p1;
    vec3 edge2 = p3 - p1;
    vec3 h = glm::cross (r.dir, edge2);
    num a = glm::dot (h, edge1);
    if (-epsilon < a && a < epsilon) // Ray is approximately parallel
    {
      return false;
    }
    num f = CONST (1.0) / a;
    vec3 s = r.orig - p1;
    num u = f * glm::dot (h, s);
    if (u < 0 || 1 < u)
    {
      return false;
    }
    vec3 q = glm::cross (s, edge1);
    num v = f * glm::dot (r.dir, q);
    if (v < CONST (0) || CONST (1) < u + v)
    {
      return false;
    }
    num t = f * glm::dot (edge2, q);
    if (t < t_min || t_max < t)
    {
      return false;
    }
    // 2. Assign proper values to hitrec
    rec.t = t;
    rec.point = r.at (t);
    rec.mat_idx = mat_idx;
    rec.set_face_normal (r, surface_normal);
    return true;
  }

  int mat_idx;
  point3 p1, p2, p3;
  vec3 surface_normal;
};

struct null_hittable : public IHittable
{

  __host__ __device__
  null_hittable ()
  {
  }

  __host__ __device__ inline bool
  hit (const ray& r, num t_min, num t_max, hit_record& rec) const override
  {
    return false;
  }
};

union hittable_data
{
  sphere s;
  plane p;
  rectangle r;
  triangle t;
  null_hittable n;

  __host__ __device__ inline hittable_data (sphere const& s) : s (s)
  {
  }
  __host__ __device__ inline hittable_data (plane const& p) : p (p)
  {
  }
  __host__ __device__ inline hittable_data (rectangle const& r) : r (r)
  {
  }
  __host__ __device__ inline hittable_data (triangle const& t) : t (t)
  {
  }

  __host__ __device__ inline hittable_data (sphere&& s) : s (std::move (s))
  {
  }
  __host__ __device__ inline hittable_data (plane&& p) : p (std::move (p))
  {
  }
  __host__ __device__ inline hittable_data (rectangle&& r) : r (std::move (r))
  {
  }
  __host__ __device__ inline hittable_data (triangle&& t) : t (std::move (t))
  {
  }

  __host__ __device__ inline hittable_data () : n ()
  {
  }

  template<typename... Args>
  __host__ __device__ inline bool
  hit (hittable_id id, Args&&... args) const
  {
    switch (id)
    {
      case hittable_id::Sphere:
        return s.hit (std::forward<Args> (args)...);
      case hittable_id::Plane:
        return p.hit (std::forward<Args> (args)...);
      case hittable_id::Rectangle:
        return r.hit (std::forward<Args> (args)...);
      case hittable_id::Triangle:
        return t.hit (std::forward<Args> (args)...);
      default:
        return n.hit (std::forward<Args> (args)...);
    }
  }
};

struct hittable
{
  __host__ __device__ inline hittable (sphere const& s)
    : id (hittable_id::Sphere), data (s)
  {
  }
  __host__ __device__ inline hittable (plane const& p)
    : id (hittable_id::Plane), data (p)
  {
  }
  __host__ __device__ inline hittable (rectangle const& r)
    : id (hittable_id::Rectangle), data (r)
  {
  }
  __host__ __device__ inline hittable (triangle const& t)
    : id (hittable_id::Triangle), data (t)
  {
  }

  __host__ __device__ inline hittable (sphere&& s)
    : id (hittable_id::Sphere), data (std::move (s))
  {
  }
  __host__ __device__ inline hittable (plane&& p)
    : id (hittable_id::Plane), data (std::move (p))
  {
  }
  __host__ __device__ inline hittable (rectangle&& r)
    : id (hittable_id::Rectangle), data (std::move (r))
  {
  }
  __host__ __device__ inline hittable (triangle&& t)
    : id (hittable_id::Triangle), data (std::move (t))
  {
  }

  __host__ __device__ inline hittable () : id (hittable_id::Unknown), data ()
  {
  }

  __host__ __device__ inline hittable (hittable const& h) : id (h.id), data ()
  {
    switch (h.id)
    {
      case hittable_id::Sphere:
        data.s = h.data.s;
        break;
      case hittable_id::Plane:
        data.p = h.data.p;
        break;
      case hittable_id::Rectangle:
        data.r = h.data.r;
        break;
      case hittable_id::Triangle:
        data.t = h.data.t;
        break;
      case hittable_id::Unknown:
        break;
    }
  }

  __host__ __device__ hittable&
  operator= (hittable const& h)
  {
    if (&h != this)
    {
      id = h.id;
      switch (h.id)
      {
        case hittable_id::Sphere:
          data.s = h.data.s;
          break;
        case hittable_id::Plane:
          data.p = h.data.p;
          break;
        case hittable_id::Rectangle:
          data.r = h.data.r;
          break;
        case hittable_id::Triangle:
          data.t = h.data.t;
          break;
        case hittable_id::Unknown:
          break;
      }
    }
    return *this;
  }

  __host__ __device__ inline bool
  hit (const ray& r, num t_min, num t_max, hit_record& rec) const
  {
    return data.hit (id, r, t_min, t_max, rec);
  }

  hittable_id id;
  hittable_data data;
};

#endif