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
  Emissive
};

class material
{
protected:
  __host__ __device__
  material (material_id i) : id (i)
  {
  }

public:
  __host__ __device__ virtual ~material() {

  }

  __host__ __device__ virtual bool
  scatter (RandomState* state, const ray& r_in, const hit_record& rec, color& attenuation,
           ray& scattered) const = 0;

  __host__ __device__ virtual color
  emit () const
  {
    return color(0, 0, 0);
  }

  // Forward declaration of size_of
  static int
  size_of (material* m);

  // Forward declaration of make_from
  __host__ __device__
  static material*
  make_from (material* old);

public:
  material_id id;
};

class lambertian : public material
{
public:
  __host__ __device__
  lambertian (const color& a) : material (material_id::Lambertian), albedo (a)
  {
  }

  __host__ __device__
  virtual ~lambertian() override {
  }

  __host__ __device__
  virtual bool
  scatter (RandomState* state, const ray& r_in, const hit_record& rec, color& attenuation,
           ray& scattered) const override
  {
    vec3 scatter_direction = rec.normal + random_in_hemisphere (state, rec.normal);
    scattered = ray (rec.point, scatter_direction);
    attenuation = albedo;
    return true;
  }

public:
  color albedo;
};

class metal : public material
{
public:
  __host__ __device__
  metal (const color& a, num f)
    : material (material_id::Metal), albedo (a), fuzz (f < 1 ? f : 1)
  {
  }

  __host__ __device__
  virtual ~metal() override {
  }

  __host__ __device__
  virtual bool
  scatter (RandomState* state, const ray& r_in, const hit_record& rec, color& attenuation,
           ray& scattered) const override
  {
    vec3 reflected =
      glm::reflect (glm::normalize (r_in.direction ()), rec.normal);
    scattered =
      ray (rec.point, reflected + fuzz * random_in_hemisphere (state, rec.normal));
    attenuation = albedo;
    return (glm::dot (scattered.direction (), rec.normal) > 0);
  }

public:
  color albedo;
  num fuzz;
};

class dielectric : public material
{
public:
  __host__ __device__
  dielectric (const color& a, num index_of_refraction)
    : material (material_id::Dielectric), albedo (a), ir (index_of_refraction)
  {
  }

  __host__ __device__
  virtual ~dielectric() override {
  }

  __host__ __device__
  virtual bool
  scatter (RandomState* state, const ray& r_in, const hit_record& rec, color& attenuation,
           ray& scattered) const override
  {
    attenuation = albedo;
    num refraction_ratio = rec.front_face ? (CONST (1) / ir) : ir;

    vec3 unit_direction = glm::normalize (r_in.direction ());
    num cos_theta = fmin (glm::dot (-unit_direction, rec.normal), 1.0);
    num sin_theta = sqrt (CONST (1) - cos_theta * cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > 1.0;
    vec3 direction;

    if (cannot_refract){
      direction = glm::reflect(unit_direction, rec.normal);
    } else if (reflectance (cos_theta, refraction_ratio) > random_positive_unit (state)) {
      direction = glm::reflect (unit_direction, rec.normal);
    } else {
      direction = glm::refract (unit_direction, rec.normal, refraction_ratio);
    }
    scattered = ray (rec.point, direction);
    return true;
  }

public:
  color albedo;
  num ir; // Index of Refraction

private:
  __host__ __device__
  static num
  reflectance (num cosine, num ref_idx)
  {
    // Use Schlick's approximation for reflectance.
    auto r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow ((1 - cosine), 5);
  }
};

class emissive : public material
{
public:
  __host__ __device__
  emissive (const color& c, const num intensity)
    : material (material_id::Emissive), emitted_color (c * intensity)
  {
  }

  __host__ __device__
  virtual ~emissive() override {
  }

  __host__ __device__
  virtual bool
  scatter (RandomState* state, const ray& r_in, const hit_record& rec, color& attenuation,
           ray& scattered) const override
  {
    return false;
  }

  __host__ __device__
  virtual color
  emit() const override
  {
    return emitted_color;
  }

public:
  color emitted_color;
};

__host__
inline
int
material::size_of (material* m)
{
  switch (m->id)
  {
    case material_id::Lambertian:
      return sizeof (lambertian);
    case material_id::Metal:
      return sizeof (metal);
    case material_id::Dielectric:
      return sizeof (dielectric);
    case material_id::Emissive:
      return sizeof (emissive);
    default:
      return 0;
  }
}

__host__ __device__
inline
material*
material::make_from (material* old)
{
  switch (old->id)
  {
    case material_id::Lambertian:
      return new lambertian (*reinterpret_cast<lambertian*> (old));
    case material_id::Metal:
      return new metal (*reinterpret_cast<metal*> (old));
    case material_id::Dielectric:
      return new dielectric (*reinterpret_cast<dielectric*> (old));
    case material_id::Emissive:
      return new emissive (*reinterpret_cast<emissive*> (old));
    default:
      return nullptr;
  }
}

#endif
