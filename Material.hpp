#pragma once

#include "Hittable.hpp"
#include "Ray.hpp"
#include "Util.hpp"
#include "types.hpp"
#include <glm/geometric.hpp>
#include <glm/glm.hpp>

struct hit_record;

enum class material_id
{
  Lambertian,
  Metal,
  Dielectric
};

class material
{
protected:
  HOST_DEVICE
  material (material_id i) : id (i)
  {
  }

public:
  HOST_DEVICE virtual bool
  scatter (RandomState* state, const ray& r_in, const hit_record& rec, color& attenuation,
           ray& scattered) const = 0;

  // Forward declaration of size_of
  static int
  size_of (material* m);

  // Forward declaration of make_from
  HOST_DEVICE
  static material*
  make_from (material* old);

public:
  material_id id;
};

class lambertian : public material
{
public:
  HOST_DEVICE
  lambertian (const color& a) : material (material_id::Lambertian), albedo (a)
  {
  }

  HOST_DEVICE
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
  HOST_DEVICE
  metal (const color& a, num f)
    : material (material_id::Metal), albedo (a), fuzz (f < 1 ? f : 1)
  {
  }

  HOST_DEVICE
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
  HOST_DEVICE
  dielectric (const color& a, num index_of_refraction)
    : material (material_id::Dielectric), albedo (a), ir (index_of_refraction)
  {
  }

  HOST_DEVICE
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
  HOST_DEVICE
  static num
  reflectance (num cosine, num ref_idx)
  {
    // Use Schlick's approximation for reflectance.
    auto r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow ((1 - cosine), 5);
  }
};

HOST
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
    default:
      return 0;
  }
}

HOST_DEVICE
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
    default:
      return nullptr;
  }
}
