#pragma once

#include <glm/vec3.hpp>
#include "types.hpp"

namespace glm
{
    inline vec3 operator*(double t, const vec3 &v)
    {
        return vec3(t * v.x, t * v.y, t * v.z);
    }
    
    inline vec3 operator*(const vec3 &v, double t)
    {
        return vec3(t * v.x, t * v.y, t * v.z);
    }

    inline vec3 operator+(double t, const vec3 &v)
    {
        return vec3(t + v.x, t + v.y, t + v.z);
    }

    inline vec3 operator-(const vec3 &v, double t)
    {
        return vec3(v.x - t, v.y - t, v.z - t);
    }

    inline vec3 operator/(const vec3 &v, double t)
    {
        return vec3(v.x / t, v.y / t, v.z / t);
    }

    inline vec3 operator*(precision t, const vec3 &v)
    {
        return vec3(t * v.x, t * v.y, t * v.z);
    }

    inline vec3 operator*(const vec3 &v, precision t)
    {
        return vec3(t * v.x, t * v.y, t * v.z);
    }

    inline vec3 operator+(precision t, const vec3 &v)
    {
        return vec3(t + v.x, t + v.y, t + v.z);
    }

    inline vec3 operator-(const vec3 &v, precision t)
    {
        return vec3(v.x - t, v.y - t, v.z - t);
    }

    inline vec3 operator/(const vec3 &v, precision t)
    {
        return vec3(v.x / t, v.y / t, v.z / t);
    }
}