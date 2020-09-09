#include <random>

#include "types.hpp"
#include "constants.hpp"

class random_gen
{
public:
    std::mt19937 gen;

    precision
    random_positive_unit()
    {
        static std::uniform_real_distribution<precision> dist(precision{0.0}, precision{1.0});
        return dist(gen);
    }

    precision
    random_unit()
    {
        static std::uniform_real_distribution<precision> dist(precision{-1.0}, precision{1.0});
        return dist(gen);
    }

    precision
    random_angle()
    {
        static std::uniform_real_distribution<precision> dist(precision{0.0}, precision{2.0 * pi});
        return dist(gen);
    }
};