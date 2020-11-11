#include <iostream>
#include <glm/glm.hpp>
#include <toml.hpp>

#include "types.hpp"
#include "constants.hpp"

#include "Camera.hpp"
#include "Image.hpp"
#include "Ray.hpp"
#include "Util.hpp"
#include "Sphere.hpp"
#include "HittableList.hpp"
#include "Material.hpp"
#include "TOMLLoader.hpp"

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

int main(int argc, char* argv[])
{
    string filename;
    if (argc == 0)
    {
        filename = "scene.toml";
    }
    else
    {
        filename = argv[1];
    }

    const auto scene_data = toml::parse(filename);

    // Image
    auto [samples_per_pixel, max_depth, image_width, image_height] = loadParams(scene_data);

    // World
    scene world = loadScene(scene_data);

    // Camera
    camera cam = loadCamera(scene_data);

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
                num u = (i + rng.random_positive_unit()) / (image_width-1);
                num v = (j + rng.random_positive_unit()) / (image_height-1);
                ray r = cam.get_ray(u, v);
                pixel_color += ray_color(r, world.objects, max_depth);
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
