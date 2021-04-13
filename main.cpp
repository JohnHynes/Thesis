#include <Eigen/Dense>
#include <Eigen/SVD>
#include <fstream>
#include <glm/glm.hpp>
#include <iostream>
#include <omp.h>
#include <toml.hpp>

#include "constants.hpp"
#include "types.hpp"

#include "Camera.hpp"
#include "Image.hpp"
#include "Material.hpp"
#include "Ray.hpp"
#include "TOMLLoader.hpp"
#include "Util.hpp"

color trace_ray(RandomState *state, ray r, color background_color,
                const scene *world, int depth) {
  hit_record rec;
  color attenuation;
  color result_color(1, 1, 1);

  while (depth > 0) {
    bool has_hit = false;
    {
      num closest_seen = infinity;
      hit_record temp_hitrec;

      for (int i = 0; i < world->object_count; ++i) {
        if (world->objects[i].hit(r, 0.0001, closest_seen, temp_hitrec)) {
          has_hit = true;
          closest_seen = temp_hitrec.t;
          rec = temp_hitrec;
        }
      }
    }
    if (has_hit) {
      if (world->materials[rec.mat_idx].scatter((RandomState *)state, r, rec,
                                                attenuation, r)) {
        result_color *= attenuation;
      } else {
        result_color *= world->materials[rec.mat_idx].emit();
        break;
      }
    } else {
      result_color *= background_color;
      break;
    }
    --depth;
  }
  return result_color;
}

void svd_denoise(Eigen::MatrixXf &image, size_t n, size_t m, float percent) {
  // Splitting image using SVD
  Eigen::JacobiSVD<Eigen::MatrixXf> SVD(image,
                                     Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::MatrixXf U = SVD.matrixU();
  Eigen::MatrixXf V = SVD.matrixV();
  Eigen::VectorXf Svec = SVD.singularValues();

  // Deleting low singular values
  for (size_t i = Svec.size() - 1; Svec.size() * percent < i; --i) {
    Svec[i] = 0;
  }

  // Reconstructing Matrix
  Eigen::MatrixXf Smat(n, m);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      if (i == j) {
        Smat(i, j) = Svec(i);
      } else {
        Smat(i, j) = 0;
      }
    }
  }

  image = U * Smat * V.transpose();
}

int main(int argc, char *argv[]) {
  string filename{"scene.toml"};
  string output{"image.ppm"};
  if (argc > 1) {
    filename = argv[1];
  }
  if (argc > 2) {
    output = argv[2];
  }
  std::cerr << "Command: " << argv[0] << ' ' << filename << ' ' << output
            << '\n';

  const auto scene_data = toml::parse(filename);

  // Image
  auto [samples_per_pixel, max_depth, image_width, image_height,
        background_color] = loadParams(scene_data);

  // World
  scene world = loadScene(scene_data);

  // Camera
  camera cam = loadCamera(scene_data);

  auto image = make_image<color>(image_width, image_height);

  RandomStateCPU cpu_state;
  RandomState *state = (RandomState *)&cpu_state;

  auto start = omp_get_wtime();

// Rendering image
#pragma omp parallel for collapse(2) schedule(guided, 16)
  for (int j = 0; j < image_height; ++j) {
    for (int i = 0; i < image_width; ++i) {
      color pixel_color(0.0, 0.0, 0.0);
      for (int s = 0; s < samples_per_pixel; ++s) {
        num u = (i + random_positive_unit(state)) / (image_width - 1);
        num v = (j + random_positive_unit(state)) / (image_height - 1);
        ray r = cam.get_ray(state, u, v);
        pixel_color += trace_ray(state, r, background_color, &world, max_depth);
      }
      image(j, i) = pixel_color;
    }
  }

  auto stop = omp_get_wtime();
  std::cout << (stop - start) << std::endl;

  // Splitting image into R G and B elements then using SVD Denoising

  float percent = 0.95;
  // Denoising Red
  Eigen::MatrixXf red_image(image_height, image_width);
  for (int i = 0; i < image_height; ++i) {
    for (int j = 0; j < image_width; ++j) {
      red_image(i, j) = image(i, j).r;
    }
  }
  svd_denoise(red_image, image_height, image_width, percent);

  // Denoising Green
  Eigen::MatrixXf green_image(image_height, image_width);
  for (int i = 0; i < image_height; ++i) {
    for (int j = 0; j < image_width; ++j) {
      green_image(i, j) = image(i, j).g;
    }
  }
  svd_denoise(green_image, image_height, image_width, percent);

  // Denoising Blue
  Eigen::MatrixXf blue_image(image_height, image_width);
  for (int i = 0; i < image_height; ++i) {
    for (int j = 0; j < image_width; ++j) {
      blue_image(i, j) = image(i, j).b;
    }
  }
  svd_denoise(blue_image, image_height, image_width, percent);

  // Reconstructing image

  for (int i = 0; i < image_height; ++i) {
    for (int j = 0; j < image_width; ++j) {
      image(i, j).r = red_image(i, j);
      image(i, j).g = green_image(i, j);
      image(i, j).b = blue_image(i, j);
    }
  }

  // Outputting Render Data
  std::ofstream ofs{output};
  ofs << "P3\n";
  ofs << image_width << " " << image_height << "\n";
  ofs << 255 << "\n";

  for (int j = image_height - 1; j >= 0; --j) {
    for (int i = 0; i < image_width; ++i) {
      auto pixel_color = image(j, i);
      write_color(ofs, pixel_color, samples_per_pixel);
    }
  }
}
