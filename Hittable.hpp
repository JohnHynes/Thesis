#pragma once

#include "types.hpp"
#include "Ray.hpp"

class material;

struct hit_record
{
    point3 point;
    vec3 normal;
    material* mat_ptr;
    num t;
    bool front_face;

    inline void set_face_normal(const ray &r, const vec3 &outward_normal)
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
    virtual bool hit(const ray &r, num tmin, num tmax, hit_record &hitrec) const = 0;
};