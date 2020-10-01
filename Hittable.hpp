#pragma once

#include <memory>

#include "types.hpp"
#include "Ray.hpp"

class material;

struct hit_record
{
    point3 point;
    glm::vec3 normal;
    std::shared_ptr<material> mat_ptr;
    precision t;
    bool front_face;

    inline void set_face_normal(const ray &r, const glm::vec3 &outward_normal)
    {
        if (glm::dot(r.dir, outward_normal) < 0)
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