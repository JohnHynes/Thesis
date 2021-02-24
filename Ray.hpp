#ifndef GPU_RAY_TRACING_RAY_HPP_
#define GPU_RAY_TRACING_RAY_HPP_

#include "Preprocessor.hpp"

#include "types.hpp"

#include <glm/vec3.hpp>

class ray
{
public:
    point3 orig;
    vec3 dir;

public:
    // Constructors
    __host__ __device__
    inline
    ray() : orig(), dir() {}
    
    constexpr ray(ray const &) = default;
    constexpr ray(ray &&) = default;
    ray &operator=(ray const &) = default;
    ray &operator=(ray &&) = default;

    __host__ __device__
    inline
    ray(const point3 &neworigin, const vec3 &newdir)
        : orig(neworigin), dir(newdir) {}

    __host__ __device__
    inline
    point3 origin() const  { return orig; }
    
    __host__ __device__
    inline
    vec3 direction() const { return dir; }

    // Member Functions
    __host__ __device__
    inline
    point3 at(num t) const
    {
        return orig + t * dir;
    }
};

#endif
