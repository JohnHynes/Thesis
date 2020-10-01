#pragma once

#include "types.hpp"
#include "Ray.hpp"
#include "Util.hpp"

struct hit_record;

class material
{
public:
    virtual bool scatter(
        const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const = 0;
};

class lambertian : public material
{
public:
    color albedo;

public:
    lambertian(const color &a) : albedo(a) {}

    virtual bool scatter(
        const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const override
    {
        (void)r_in;
        glm::vec3 scatter_direction = random_in_hemisphere(rec.normal);
        scattered = ray(rec.point, scatter_direction);
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
    metal(const color &a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    virtual bool scatter(
        const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const override
    {
        glm::vec3 reflected = reflect(glm::normalize(r_in.dir), rec.normal);
        scattered = ray(rec.point, reflected + fuzz * random_in_hemisphere(rec.normal));
        attenuation = albedo;
        return true;
    }
};