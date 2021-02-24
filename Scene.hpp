#ifndef GPU_RAY_TRACING_SCENE_HPP_
#define GPU_RAY_TRACING_SCENE_HPP_

#include "Preprocessor.hpp"

#include <string>

#include "Hittable.hpp"
#include "Material.hpp"
#include "types.hpp"

#ifdef USE_GPU
class scene;

__global__ void
fixup (scene* s, material** materials, hittable** objects);
#endif

class scene
{
public:
  hittable** objects;
  int object_count;
  material** materials;
  int material_count;

  void
  free_host ()
  {
    for (int i = 0; i < object_count; ++i)
    {
      delete objects[i];
    }
    for (int i = 0; i < material_count; ++i)
    {
      delete materials[i];
    }
    delete objects;
    delete materials;
  }

#ifdef USE_GPU

  scene*
  copy_to_device ()
  {
    scene* d_scene;
    // make space for the scene
    CUDA_CALL(cudaMalloc (&d_scene, sizeof (scene)));
    // copy the sizes lazily -- the pointers are invalid but we are overwriting them
    CUDA_CALL(cudaMemcpy (d_scene, this, sizeof (scene), cudaMemcpyHostToDevice));

    hittable** d_objects;
    {
      hittable** gpu_objects = new hittable*[object_count];
      for (int i = 0; i < object_count; ++i)
      {
        int size = hittable::size_of (objects[i]);
        CUDA_CALL(cudaMalloc (&gpu_objects[i], size));
        CUDA_CALL(cudaMemcpy (gpu_objects[i], objects[i], size, cudaMemcpyHostToDevice));
      }
      const int size = sizeof(hittable*) * object_count;
      CUDA_CALL(cudaMalloc (&d_objects, size));
      CUDA_CALL(cudaMemcpy (d_objects, gpu_objects, size, cudaMemcpyHostToDevice));
      delete[] gpu_objects;
    }

    material** d_materials;
    {
      material** gpu_materials = new material*[material_count];
      for (int i = 0; i < material_count; ++i)
      {
        int size = material::size_of (materials[i]);
        CUDA_CALL(cudaMalloc (&gpu_materials[i], size));
        CUDA_CALL(cudaMemcpy (gpu_materials[i], materials[i], size, cudaMemcpyHostToDevice));
      }
      const int size = sizeof(material*) * material_count;
      CUDA_CALL(cudaMalloc (&d_materials, size));
      CUDA_CALL(cudaMemcpy (d_materials, gpu_materials, size, cudaMemcpyHostToDevice));
      delete[] gpu_materials;
    }

    ////////////////////////////////////////////////////////////

    int max = std::max (material_count, object_count);
    int const threads_per_block = 64;
    int const blocks = (max + threads_per_block - 1) / threads_per_block;
    fixup<<<blocks, threads_per_block>>> (d_scene, d_materials, d_objects);
    CUDA_CALL(cudaDeviceSynchronize());
    return d_scene;
  }

/*
  void
  free_device ()
  {
    for (int i = 0; i < object_count; ++i)
    {
      cudaFree (objects[i]);
    }
    for (int i = 0; i < material_count; ++i)
    {
      cudaFree (materials[i]);
    }
    cudaFree (objects);
    cudaFree (materials);
  }
  */
#endif
};

#ifdef __CUDACC__

// TODO: move to Scene.cpp

__global__ void
fixup (scene* s,  material** materials, hittable** objects)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int M = s->material_count;
  const int O = s->object_count;
  s->materials = materials;
  s->objects = objects;
  if (i < M)
  {
    // force v-table creation on the GPU
    material* new_m = material::make_from (s->materials[i]);
    // free the CPU version that was copied from the GPU
    free (s->materials[i]);
    // reassign material pointer to be GPU-created material
    s->materials[i] = new_m;
  }
  if (i < O)
  {
    // force v-table creation on the GPU
    hittable* new_h = hittable::make_from (s->objects[i]);
    // free the CPU version that was copied from the GPU
    free (s->objects[i]);
    // reassign hittable pointer to be GPU-created hittable
    s->objects[i] = new_h;
  }
}
#endif

#endif
