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
        glm::vec3 scatter_direction = rec.normal + random_unit_vector();
        scattered = ray(rec.point, scatter_direction);
        attenuation = albedo;
        return true;
    }
};

class metal : public material
{
public:
    color albedo;

public:
    metal(const color &a) : albedo(a) {}

    virtual bool scatter(
        const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const override
    {
        glm::vec3 reflected = reflect(glm::normalize(r_in.dir), rec.normal);
        scattered = ray(rec.point, reflected);
        attenuation = albedo;
        return (glm::dot(scattered.dir, rec.normal) > 0);
    }
};