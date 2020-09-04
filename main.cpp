#include <iostream>
#include <glm/glm.hpp>
#include <glm/ext/matrix_transform.hpp>

#include "Ray.hpp"
#include "Util.hpp"
#include "Sphere.hpp"
#include "HittableList.hpp"

constexpr glm::vec3 origin(0.0f, 0.0f, 0.0f);

glm::vec3 ray_color(const ray& r, const hittable_list& world)
{
    hit_record hitrec;
    if (world.hit(r, 0, 20, hitrec))
    {
        return 0.5f * (hitrec.normal + glm::vec3(1, 1, 1));
    }
    return blend_color(r);
}

int main()
{
    const glm::vec3 origin(0.0f, 0.0f, 0.0f);

    // Image Size
    constexpr double aspect_ratio = 16.0 / 9;
    constexpr int image_width = 1080;
    constexpr int image_height = static_cast<int>(image_width / aspect_ratio);

    // World
    hittable_list world;
    world.add(std::make_shared<sphere>(glm::vec3(0, 0, -1), 0.5));
    world.add(std::make_shared<sphere>(glm::vec3(0.25, 0.25, 0.6), 0.35));

    // Camera
    float viewport_height = 2.0;
    float viewport_width = aspect_ratio * viewport_height;
    float focal_length = 1.0;

    glm::vec3 horizontal = glm::vec3(viewport_width, 0.0f, 0.0f);
    glm::vec3 vertical = glm::vec3(0.0f, viewport_height, 0.0f);
    glm::vec3 lower_left_corner = origin - horizontal / 2.0f - vertical / 2.0f - glm::vec3(0.0f, 0.0f, focal_length);

    // Render
    std::cout << "P3\n";
    std::cout << image_width << " " << image_height << "\n";
    std::cout << 255 << "\n";

    for (int j = image_height - 1; j >= 0; --j)
    {
        for (int i = 0; i < image_width; ++i)
        {
            float u = float(i) / (image_width - 1);
            float v = float(j) / (image_height - 1);
            ray r(origin, lower_left_corner + u * horizontal + v * vertical);
            printAsColor(ray_color(r, world));
        }
    }
}