#include <iostream>
#include <glm/glm.hpp>
#include <glm/ext/matrix_transform.hpp>

#include "Camera.hpp"
#include "Ray.hpp"
#include "Util.hpp"
#include "Sphere.hpp"
#include "HittableList.hpp"

constexpr glm::vec3 origin(0.0f, 0.0f, 0.0f);

glm::vec3 ray_color(const ray& r, const hittable_list& world)
{
    hit_record hitrec;
    if (world.hit(r, 0, 100, hitrec))
    {
        return 0.5f * (hitrec.normal + glm::vec3(1, 1, 1));
    }
    return blend_color(r);
}

int main()
{
    const glm::vec3 origin(0.0f, 0.0f, 0.0f);

    // Image
    // 1920x1080
    // 3840x2160
    constexpr double aspect_ratio = 16.0 / 10.0;
    constexpr int image_width = 3840;
    constexpr int image_height = static_cast<int>(image_width / aspect_ratio);
    constexpr int samples_per_pixel = 100;

    // World
    hittable_list world;
    world.add(std::make_shared<sphere>(glm::vec3(0, 0, -1), 0.5));
    world.add(std::make_shared<sphere>(glm::vec3(0, -100.5, -1), 100));

    // Camera
    camera cam;

    // Render
    std::cout << "P3\n";
    std::cout << image_width << " " << image_height << "\n";
    std::cout << 255 << "\n";

    for (int j = image_height - 1; j >= 0; --j)
    {
        for (int i = 0; i < image_width; ++i)
        {
            glm::vec3 pixel_color(0.0f, 0.0f, 0.0f);
            for (int s = 0; s < samples_per_pixel; ++s)
            {
                float u = (i + random_float()) / (image_width - 1);
                float v = (j + random_float()) / (image_height - 1);
                ray r = cam.get_ray(u, v);
                pixel_color += ray_color(r, world);
            }
            write_color(std::cout, pixel_color, samples_per_pixel);
        }
    }
}