#include <random>
#include <glm/vec3.hpp>

#include "types.hpp"
#include "constants.hpp"

class random_gen
{
public:
    std::mt19937 gen;

    num
    random_positive_unit()
    {
        thread_local std::uniform_real_distribution<num> dist(num{0.0}, num{1.0});
        return dist(gen);
    }

    num
    random_unit()
    {
        thread_local std::uniform_real_distribution<num> dist(num{-1.0}, num{1.0});
        return dist(gen);
    }

    num
    random_angle()
    {
        thread_local std::uniform_real_distribution<num> dist(num{0.0}, num{2.0 * pi});
        return dist(gen);
    }

    int
    random_int(int low, int high)
    {
        return static_cast<int>(random_positive_unit() * (high - low) + low);
    }

    vec3
    random_unit_vector ()
    {
        num a = random_angle ();
        num z = random_unit ();
        num r = sqrt (num(1) - z * z);
        return vec3 (r * cos (a), r * sin (a), z);
    }

    vec3 random_in_unit_sphere() {
        while (true) {
            vec3 p = vec3(random_unit(), random_unit(), random_unit());
            if (glm::dot(p, p) >= 1) continue;
            return p;
        }
    }

    vec3
    random_in_hemisphere (const vec3 &normal)
    {
        vec3 in_unit_sphere = random_unit_vector ();
        if (glm::dot (in_unit_sphere, normal) > 0.0)
            return in_unit_sphere;
        else
            return -in_unit_sphere;
    }
};