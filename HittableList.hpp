#pragma once

#include "Hittable.hpp"

#include <memory>
#include <vector>

class hittable_list : public hittable
{
public:
    std::vector<std::shared_ptr<hittable>> objects;

public:
    // Constructors
    hittable_list(){};
    hittable_list(hittable_list const &) = default;
    hittable_list(hittable_list &&) = default;
    hittable_list &operator=(hittable_list const &) = default;
    hittable_list &operator=(hittable_list &&) = default;

    hittable_list(std::shared_ptr<hittable> object) { add(object); };

    // Mutators
    void clear() { objects.clear(); }
    void add(std::shared_ptr<hittable> object) { objects.push_back(object); }

    // Member Functions
    bool hit(const ray &r, float tmin, float tmax, hit_record &hitrec) const
    {
        hit_record temp_hitrec;
        bool has_hit = false;
        float closest_seen = tmax;

        for (const auto &object : objects)
        {
            if (object->hit(r, tmin, closest_seen, temp_hitrec))
            {
                has_hit = true;
                closest_seen = temp_hitrec.t;
                hitrec = temp_hitrec;
            }
        }

        return has_hit;
    }
};