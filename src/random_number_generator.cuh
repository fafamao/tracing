#ifndef RANDOM_NUMBER_GENERATOR_CUH_
#define RANDOM_NUMBER_GENERATOR_CUH_

#include <curand_kernel.h>

extern __device__ curandState *render_rand_state_global;

__global__ void rand_init(curandState *rand_state);
__global__ void render_init(int max_x, int max_y);


#endif // RANDOM_NUMBER_GENERATOR_CUH_