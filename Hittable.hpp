#pragma once

#include "types.hpp"
#include "Ray.hpp"

struct hit_record
{
    glm::vec3 point;
    glm::vec3 normal;
    precision t;
    bool front_face;

    inline void set_face_normal(const ray &r, const glm::vec3 &outward_normal)
    {
        front_face = glm::dot(r.dir, outward_normal) < 0;
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

class hittable
{
public:
    virtual bool hit(const ray &r, precision tmin, precision tmax, hit_record &hitrec) const = 0;
};