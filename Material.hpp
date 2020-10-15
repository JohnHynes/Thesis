#pragma once

#include "Ray.hpp"
#include "Util.hpp"
#include <glm/glm.hpp>
#include "types.hpp"
#include <glm/geometric.hpp>

struct hit_record;

class material
{
public:
  virtual bool
  scatter (const ray &r_in, const hit_record &rec, color &attenuation,
           ray &scattered) const = 0;
};

class lambertian : public material
{
public:
  color albedo;

public:
  lambertian (const color &a) : albedo (a)
  {
  }

  virtual bool
  scatter (const ray &r_in, const hit_record &rec, color &attenuation,
           ray &scattered) const override
  {
    (void)r_in;
    vec3 scatter_direction = random_in_hemisphere (rec.normal);
    scattered = ray (rec.point, scatter_direction);
    attenuation = albedo;
    return true;
  }
};

class metal : public material
{
public:
  color albedo;
  precision fuzz;

public:
  metal (const color &a, double f) : albedo (a), fuzz (f < 1 ? f : 1)
  {
  }

  virtual bool
  scatter (const ray &r_in, const hit_record &rec, color &attenuation,
           ray &scattered) const override
  {
    vec3 reflected = reflect (glm::normalize (r_in.dir), rec.normal);
    scattered =
      ray (rec.point, reflected + fuzz * random_in_hemisphere (rec.normal));
    attenuation = albedo;
    return true;
  }
};

class dielectric : public material
{
public:
  double ir;

public:
  dielectric (precision index_of_refraction) : ir (index_of_refraction)
  {
  }

  virtual bool
  scatter (const ray &r_in, const hit_record &rec, color &attenuation,
           ray &scattered) const override
  {
    attenuation = color(0.9f, 0.9f, 0.9f);
    precision refraction_ratio = rec.front_face ? (CONST(1.0) / ir) : ir;
    vec3 unit_direction = glm::normalize(r_in.dir);
    precision cos_theta = glm::min(glm::dot(-unit_direction, rec.normal), CONST(1.0));
    precision sin_theta = glm::sqrt(CONST(1.0) - cos_theta * cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > CONST(1.0);
    vec3 direction;
    if (cannot_refract)
        direction = glm::reflect(unit_direction, rec.normal);
    else
        direction = glm::refract(unit_direction, rec.normal, refraction_ratio);
    
    scattered = ray(rec.point, direction);
    return true;
  }

};