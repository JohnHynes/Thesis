#pragma once

#include "Ray.hpp"
#include "Hittable.hpp"

class sphere : public hittable
{
public:
    glm::vec3 center;
    precision radius;

public:
    // Constructors
    sphere() = default;
    constexpr sphere(sphere const &) = default;
    constexpr sphere(sphere &&) = default;
    constexpr sphere &operator=(sphere const &) = default;
    constexpr sphere &operator=(sphere &&) = default;

    sphere(const glm::vec3& newcenter, precision newradius) : center(newcenter), radius(newradius) {}

    // Member Functions
    bool hit(const ray &r, precision tmin, precision tmax, hit_record &hitrec) const
    {
        glm::vec3 oc = glm::vec3(0.0f, 0.0f, 0.0f) - center;
        precision a = glm::dot(r.dir, r.dir);
        precision b = glm::dot(oc, r.dir);
        precision c = glm::dot(oc, oc) - radius * radius;
        precision discriminant = b * b - a * c;

        if (discriminant > 0)
        {
            precision root = sqrt(discriminant);

            precision t = (-b - root) / a;
            if (tmin < t && t < tmax)
            {
                hitrec.t = t;
                hitrec.point = r.at(hitrec.t);
                hitrec.normal = (hitrec.point - center) / radius;
                glm::vec3 outward_normal = (hitrec.point - center) / radius;
                hitrec.set_face_normal(r, outward_normal);
                return true;
            }

            t = (-b + root) / a;
            if (tmin < t && t < tmax)
            {
                hitrec.t = t;
                hitrec.point = r.at(hitrec.t);
                hitrec.normal = (hitrec.point - center) / radius;
                glm::vec3 outward_normal = (hitrec.point - center) / radius;
                hitrec.set_face_normal(r, outward_normal);
                return true;
            }
        }
        return false;
    }
};