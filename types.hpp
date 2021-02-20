#pragma once

#include <glm/fwd.hpp>
#include <glm/vec3.hpp>

using num = float;
using vec3 = glm::f32vec3;
using color = vec3;
using point3 = vec3;

#define CONST(x) static_cast<num> (x)

#ifdef USE_GPU
#include <cuda.h>
#include <stdlib.h>
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    exit(-1);}} while(0)
#endif

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#define DEVICE __device__
#define HOST __host__
#else
#define HOST_DEVICE
#define DEVICE
#define HOST
#endif
