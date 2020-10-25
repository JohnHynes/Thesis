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
            return color(0, 0, 0);
    }

    num t = 0.5 * (glm::normalize(r.dir).y + 1.0);
    return glm::mix(color(1.0, 1.0, 1.0), color(0.5, 0.7, 1.0), t);
}

int main()
{
    // Image
    // 1920x1080
    // 3840x2160
    constexpr double aspect_ratio = 16.0 / 10.0;
    constexpr int image_width = 600;
    constexpr int image_height = static_cast<int>(image_width / aspect_ratio);
    constexpr int samples_per_pixel = 1000;
    constexpr int max_depth = 50;

    // World
    hittable_list world;
 
    auto material_ground = std::make_shared<lambertian>(color(0.2, 0.6, 0.2));
    auto material_red = std::make_shared<lambertian>(color(0.7, 0.2, 0.2));
    auto material_glass = std::make_shared<dielectric>(1.5);
    auto material_metal  = std::make_shared<metal>(color(0.6, 0.6, 0.8), 0.75);

    world.add(std::make_shared<sphere>(point3( 0.0, -100.5, -1.0), 100.0, material_ground));
    world.add(std::make_shared<sphere>(point3( 1.1,    0.0, -1.0),   0.5, material_glass));
    world.add(std::make_shared<sphere>(point3( 0.0,    0.0, -1.0),   0.5, material_red));
    world.add(std::make_shared<sphere>(point3(-1.1,    0.0, -1.0),   0.5, material_glass));
    // Camera
    point3 lookfrom(-2,2,1);
    point3 lookat(0,0,-1);
    glm::vec3 upv(0,1,0);
    num aperture = 0.01;
    num dist_to_focus = glm::length(lookfrom - lookat);

    camera cam(lookfrom, lookat, upv, 20, aspect_ratio, aperture, dist_to_focus);

    // Render
    std::cout << "P3\n";
    std::cout << image_width << " " << image_height << "\n";
    std::cout << 255 << "\n";

    auto image = make_image<color>(image_width, image_height);

    // Rendering image
    for (int j = 0; j < image_height; ++j)
    {
        for (int i = 0; i < image_width; ++i)
        {
            color pixel_color(0.0, 0.0, 0.0);
            for (int s = 0; s < samples_per_pixel; ++s)
            {
                num u = (i + rng.random_positive_unit()) / (image_width-1);
                num v = (j + rng.random_positive_unit()) / (image_height-1);
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
