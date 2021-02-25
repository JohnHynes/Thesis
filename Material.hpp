#ifndef GPU_RAY_TRACER_MATERIAL_HPP_
#define GPU_RAY_TRACER_MATERIAL_HPP_

#include "Preprocessor.hpp"
#include "Random.hpp"

#include "Hittable.hpp"
#include "Ray.hpp"
#include "Util.hpp"
#include "types.hpp"

#include <glm/geometric.hpp>

struct hit_record;

enum class material_id
{
  Lambertian,
  Metal,
  Dielectric,
  Emissive,
  None
};

struct IMaterial
{
  __host__ __device__ virtual bool
  scatter (RandomState* state, const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const = 0;
};

struct lambertian : public IMaterial
{
  __host__ __device__ inline lambertian (const color& a) : albedo (a)
  {
  }

  __host__ __device__ inline bool
  scatter (RandomState* state, const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override
  {
    vec3 scatter_direction =
      rec.normal + random_in_hemisphere (state, rec.normal);
    scattered = ray (rec.point, scatter_direction);
    attenuation = albedo;
    return true;
  }
  color albedo;
};

struct metal : public IMaterial
{
  __host__ __device__ inline metal (const color& a, num f)
    : albedo (a), fuzz (fminf (f, CONST (1)))
  {
  }

  __host__ __device__ inline bool
  scatter (RandomState* state, const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override
  {
    vec3 reflected =
      glm::reflect (glm::normalize (r_in.direction ()), rec.normal);
    scattered = ray (
      rec.point, reflected + fuzz * random_in_hemisphere (state, rec.normal));
    attenuation = albedo;
    return (glm::dot (scattered.direction (), rec.normal) > CONST (0));
  }

  color albedo;
  num fuzz;
};

struct dielectric : public IMaterial
{
  __host__ __device__ inline dielectric (const color& a, num index_of_refraction)
    : albedo (a), ir (index_of_refraction)
  {
  }

  __host__ __device__ inline bool
  scatter (RandomState* state, const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override
  {
    attenuation = albedo;
    num refraction_ratio = rec.front_face ? (CONST (1) / ir) : ir;

    vec3 unit_direction = glm::normalize (r_in.direction ());
    num cos_theta = fminf (glm::dot (-unit_direction, rec.normal), CONST (1));
    num sin_theta = sqrtf (CONST (1) - cos_theta * cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > CONST (1);
    vec3 direction;

    if (cannot_refract)
    {
      direction = glm::reflect (unit_direction, rec.normal);
    }
    else if (reflectance (cos_theta, refraction_ratio) > random_positive_unit (state))
    {
      direction = glm::reflect (unit_direction, rec.normal);
    }
    else
    {
      direction = glm::refract (unit_direction, rec.normal, refraction_ratio);
    }
    scattered = ray (rec.point, direction);
    return true;
  }

  color albedo;
  num ir; // Index of Refraction

private:
  __host__ __device__ static inline num
  reflectance (num cosine, num ref_idx)
  {
    // Use Schlick's approximation for reflectance.
    num r0 = (CONST (1) - ref_idx) / (CONST (1) + ref_idx);
    r0 = r0 * r0;
    return r0 + (CONST (1) - r0) * powf ((CONST (1) - cosine), 5);
  }
};

struct emissive : public IMaterial
{
  __host__ __device__ inline emissive (const color& c, const num intensity)
    : emitted_color (c * intensity)
  {
  }

  __host__ __device__ inline bool
  scatter (RandomState* state, const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override
  {
    return false;
  }

  color emitted_color;
};

struct null_material : public IMaterial
{
  __host__ __device__ inline null_material ()
  {
  }

  __host__ __device__ inline bool
  scatter (RandomState* state, const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override
  {
    return false;
  }
};

union material_data
{
  lambertian l;
  metal m;
  dielectric d;
  emissive e;
  null_material n;

  __host__ __device__ inline material_data (lambertian const& l) : l (l)
  {
  }
  __host__ __device__ inline material_data (metal const& m) : m (m)
  {
  }
  __host__ __device__ inline material_data (dielectric const& d) : d (d)
  {
  }
  __host__ __device__ inline material_data (emissive const& e) : e (e)
  {
  }

  __host__ __device__ inline material_data (lambertian&& l) : l (std::move (l))
  {
  }
  __host__ __device__ inline material_data (metal&& m) : m (std::move (m))
  {
  }
  __host__ __device__ inline material_data (dielectric&& d) : d (std::move (d))
  {
  }
  __host__ __device__ inline material_data (emissive&& e) : e (std::move (e))
  {
  }

  __host__ __device__ inline material_data () : n ()
  {
  }

  template<typename... Args>
  __host__ __device__ inline bool
  scatter (material_id id, Args&&... args) const
  {
    switch (id)
    {
      case material_id::Lambertian:
        return l.scatter (std::forward<Args> (args)...);
      case material_id::Metal:
        return m.scatter (std::forward<Args> (args)...);
      case material_id::Dielectric:
        return d.scatter (std::forward<Args> (args)...);
      case material_id::Emissive:
        return e.scatter (std::forward<Args> (args)...);
      default:
        return n.scatter (std::forward<Args> (args)...);
    }
  }

  __host__ __device__ inline color
  emit (material_id id) const
  {
    switch (id)
    {
      case material_id::Emissive:
        return e.emitted_color;
      default:
        return color{0, 0, 0};
    }
  }
};

struct material
{

  __host__ __device__ inline material (lambertian const& l)
    : id (material_id::Lambertian), data (l)
  {
  }
  __host__ __device__ inline material (metal const& m)
    : id (material_id::Metal), data (m)
  {
  }
  __host__ __device__ inline material (dielectric const& d)
    : id (material_id::Dielectric), data (d)
  {
  }
  __host__ __device__ inline material (emissive const& e)
    : id (material_id::Emissive), data (e)
  {
  }

  __host__ __device__ inline material (lambertian&& l)
    : id (material_id::Lambertian), data (std::move (l))
  {
  }
  __host__ __device__ inline material (metal&& m)
    : id (material_id::Metal), data (std::move (m))
  {
  }
  __host__ __device__ inline material (dielectric&& d)
    : id (material_id::Dielectric), data (std::move (d))
  {
  }
  __host__ __device__ inline material (emissive&& e)
    : id (material_id::Emissive), data (std::move (e))
  {
  }

  __host__ __device__ inline material () : id (material_id::None), data ()
  {
  }

  __host__ __device__ inline material (material const& m) : id (m.id), data ()
  {
    switch (m.id)
    {
      case material_id::Lambertian:
        data.l = m.data.l;
        break;
      case material_id::Metal:
        data.m = m.data.m;
        break;
      case material_id::Dielectric:
        data.d = m.data.d;
        break;
      case material_id::Emissive:
        data.e = m.data.e;
        break;
      case material_id::None:
        break;
    }
  }

  __host__ __device__ material&
  operator= (material const& m)
  {
    if (&m != this)
    {
      id = m.id;
      switch (m.id)
      {
        case material_id::Lambertian:
          data.l = m.data.l;
          break;
        case material_id::Metal:
          data.m = m.data.m;
          break;
        case material_id::Dielectric:
          data.d = m.data.d;
          break;
        case material_id::Emissive:
          data.e = m.data.e;
          break;
        case material_id::None:
          break;
      }
    }
    return *this;
  }

  __host__ __device__ inline bool
  scatter (RandomState* state, const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const
  {
    return data.scatter (id, state, r_in, rec, attenuation, scattered);
  }

  __host__ __device__ inline color
  emit () const
  {
    return data.emit (id);
  }

public:
  material_id id;
  material_data data;
};

#endif
