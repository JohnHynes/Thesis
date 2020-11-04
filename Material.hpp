#pragma once

#include "Ray.hpp"
#include "Util.hpp"
#include <glm/glm.hpp>
#include "types.hpp"
#include <glm/geometric.hpp>

struct hit_record;


class material {
    public:
        virtual bool scatter(
            const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered
        ) const = 0;
};


class lambertian : public material {
    public:
        lambertian(const color& a) : albedo(a) {}

        virtual bool scatter(
            const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered
        ) const override {
            vec3 scatter_direction = rec.normal + rng.random_in_hemisphere (rec.normal);
            scattered = ray(rec.point, scatter_direction);
            attenuation = albedo;
            return true;
        }

    public:
        color albedo;
};


class metal : public material {
    public:
        metal(const color& a, num f) : albedo(a), fuzz(f < 1 ? f : 1) {}

        virtual bool scatter(
            const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered
        ) const override {
            vec3 reflected = glm::reflect(glm::normalize(r_in.direction()), rec.normal);
            scattered = ray(rec.point, reflected + fuzz * rng.random_in_hemisphere (rec.normal));
            attenuation = albedo;
            return (glm::dot(scattered.direction(), rec.normal) > 0);
        }

    public:
        color albedo;
        num fuzz;
};


class dielectric : public material {
    public:
        dielectric(const color& a, num index_of_refraction) : albedo(a), ir(index_of_refraction) {}

        virtual bool scatter(
            const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered
        ) const override {
            attenuation = albedo;
            num refraction_ratio = rec.front_face ? (CONST(1) / ir) : ir;

            vec3 unit_direction = glm::normalize(r_in.direction());
            num cos_theta = fmin(glm::dot(-unit_direction, rec.normal), 1.0);
            num sin_theta = sqrt(CONST(1) - cos_theta*cos_theta);

            bool cannot_refract = refraction_ratio * sin_theta > 1.0;
            vec3 direction;

            if (cannot_refract || reflectance(cos_theta, refraction_ratio) > rng.random_positive_unit())
                direction = glm::reflect(unit_direction, rec.normal);
            else
                direction = glm::refract(unit_direction, rec.normal, refraction_ratio);

            scattered = ray(rec.point, direction);
            return true;
        }

    public:
        color albedo;
        num ir; // Index of Refraction

    private:
        static num reflectance(num cosine, num ref_idx) {
            // Use Schlick's approximation for reflectance.
            auto r0 = (1-ref_idx) / (1+ref_idx);
            r0 = r0*r0;
            return r0 + (1-r0)*pow((1 - cosine),5);
        }
};