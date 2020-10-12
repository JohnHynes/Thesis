#pragma once

#include "Ray.hpp"
#include "Util.hpp"
#include "glm/glm.hpp"
#include "types.hpp"

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
    glm::vec3 scatter_direction = random_in_hemisphere (rec.normal);
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
    glm::vec3 reflected = reflect (glm::normalize (r_in.dir), rec.normal);
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
    attenuation = color (1.0, 1.0, 1.0);
    precision refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

    glm::vec3 unit_direction = glm::normalize (r_in.dir);
    glm::vec3 direction = refract (unit_direction, rec.normal, refraction_ratio);

    scattered = ray(rec.point, direction);
    return true;

    /*
    precision cos_theta = fmin (glm::dot (-unit_direction, rec.normal), 1.0);
    precision sin_theta = sqrt (1.0 - cos_theta * cos_theta);

    glm::vec3 direction;
    bool cannot_refract = refraction_ratio * sin_theta > 1.0;
    //||reflectance (cos_theta, refraction_ratio) > rng.random_positive_unit ()
    if (cannot_refract)
    {
      direction = reflect (unit_direction, rec.normal);
    }
    else
    {
      direction = refract (unit_direction, rec.normal, refraction_ratio);
    }
    scattered = ray (rec.point, direction);
    return true;
    */
  }

private:
  static precision
  reflectance (precision cos, precision ref_idx)
  {
    precision r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow ((1 - cos), 5);
  }
};