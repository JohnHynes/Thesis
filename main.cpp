#include <iostream>
#include <glm/glm.hpp>
#include <glm/ext/matrix_transform.hpp>
#include "Ray.hpp"
#include "Util.hpp"

int main() {
    const glm::vec3 origin(0.0f, 0.0f, 0.0f);

    // Image Size
    constexpr double aspect_ratio = 16.0 / 9;
    constexpr int image_width = 800;
    constexpr int image_height = static_cast<int>(image_width / aspect_ratio);

    // Camera
    glm::vec3 cameraPos(0.0f, 0.0f, 3.0f);
    glm::vec3 cameraTarget(0.0f, 0.0f, 0.0f);
    glm::vec3 cameraDirection = glm::normalize(cameraPos - cameraTarget);

    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f); 
    glm::vec3 cameraRight = glm::normalize(glm::cross(up, cameraDirection));
    glm::vec3 cameraUp = glm::cross(cameraDirection, cameraRight);

    glm::mat4 view = glm::lookAt(cameraPos, cameraTarget, cameraUp);

    // Camera
    float viewport_height = 2.0;
    float viewport_width = aspect_ratio * viewport_height;
    float focal_length = 1.0;

    glm::vec3 horizontal = glm::vec3(viewport_width, 0.0f, 0.0f);
    glm::vec3 vertical = glm::vec3(0.0f, viewport_height, 0.0f);
    glm::vec3 lower_left_corner = origin - horizontal / 2.0f - vertical / 2.0f - glm::vec3(0.0f , 0.0f ,focal_length);

    // Render
    std::cout << "P3\n";
    std::cout << image_width << " " << image_height << "\n";
    std::cout << 255 << "\n";

    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            float u = float(i) / (image_width - 1);
            float v = float(j) / (image_height - 1);
            Ray r(origin, glm::vec3(lower_left_corner + u * horizontal + v * vertical), glm::vec3());
            printAsColor(blend_color(r));
        }
    }
}