#include <iostream>
#include <glm/glm.hpp>

#include "types.hpp"
#include "constants.hpp"

#include "Camera.hpp"
#include "Image.hpp"
#include "Ray.hpp"
#include "Util.hpp"
#include "Sphere.hpp"
#include "HittableList.hpp"
#include "Vec3.hpp"
#include "Material.hpp"

color ray_color(const ray &r, const hittable_list &world, int depth)
{
    if (depth <= 0)
    {
        return color(0.0, 0.0, 0.0);
    }

    hit_record rec;
    if (world.hit(r, 0.0001, infinity, rec))
    {
        ray scattered;
        color attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
            return attenuation * ray_color(scattered, world, depth - 1);
        else
        {
            return color(0, 0, 0);
        }
    }

    auto t = 0.5 * (glm::normalize(r.dir).y + 1.0);
    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

int main()
{
    // Image
    // 1920x1080
    // 3840x2160
    constexpr double aspect_ratio = 16.0 / 10.0;
    constexpr int image_width = 1920;
    constexpr int image_height = static_cast<int>(image_width / aspect_ratio);
    constexpr int samples_per_pixel = 100;
    constexpr int max_depth = 50;

    // World
    hittable_list world;

    auto material_ground = std::make_shared<lambertian>(color(0.2, 0.8, 0.2));
    auto material_matte = std::make_shared<lambertian>(color(0.2, 0.5, 0.8));
    auto material_metal = std::make_shared<metal>(color(0.6, 0.6, 0.6));

    world.add(std::make_shared<sphere>(point3(0, -100.5, -1), 100, material_ground));
    world.add(std::make_shared<sphere>(point3(0, 0, -1), 0.5, material_matte));
    world.add(std::make_shared<sphere>(point3(-1, 0, -1), 0.5, material_metal));

    // Camera
    camera cam;

    // Render
    std::cout << "P3\n";
    std::cout << image_width << " " << image_height << "\n";
    std::cout << 255 << "\n";

    auto image = make_image<color>(image_width, image_height);

// Rendering image
#pragma omp parallel for collapse(2)
    for (int j = 0; j < image_height; ++j)
    {
        for (int i = 0; i < image_width; ++i)
        {
            color pixel_color(0.0, 0.0, 0.0);
            for (int s = 0; s < samples_per_pixel; ++s)
            {
                precision u = (i + rng.random_unit()) / (image_width - 1);
                precision v = (j + rng.random_unit()) / (image_height - 1);
                ray r = cam.get_ray(u, v);
                pixel_color += ray_color(r, world, max_depth);
            }
            image(j, i) = pixel_color;
        }
    }

    // Outputting image
    for (int j = image_height - 1; j >= 0; --j)
    {
        for (int i = 0; i < image_width; ++i)
        {
            auto pixel_color = image(j, i);
            write_color(std::cout, pixel_color, samples_per_pixel);
        }
    }
}
