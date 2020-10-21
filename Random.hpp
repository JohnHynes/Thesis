#include <random>

#include "types.hpp"
#include "constants.hpp"

class random_gen
{
public:
    std::mt19937 gen;

    num
    random_positive_unit()
    {
        static std::uniform_real_distribution<num> dist(num{0.0}, num{1.0});
        return dist(gen);
    }

    num
    random_unit()
    {
        static std::uniform_real_distribution<num> dist(num{-1.0}, num{1.0});
        return dist(gen);
    }

    num
    random_angle()
    {
        static std::uniform_real_distribution<num> dist(num{0.0}, num{2.0 * pi});
        return dist(gen);
    }
};