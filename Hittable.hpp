#pragma once

#include "Ray.hpp"

struct hit_record
{
    glm::vec3 point;
    glm::vec3 normal;
    float t;
    bool front_face;

    inline void set_face_normal(const ray& r, const glm::vec3& outward_normal) {
        front_face = glm::dot(r.dir, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable
{
public:
    virtual bool hit(const ray &r, float tmin, float tmax, hit_record &hitrec) const = 0;
};