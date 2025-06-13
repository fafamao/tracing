#ifndef RANDOM_NUMBER_GENERATOR_CUH_
#define RANDOM_NUMBER_GENERATOR_CUH_

#include <curand_kernel.h>

__global__ void rand_init(curandState *rand_state);
__global__ void render_init(int max_x, int max_y, curandState *rand_state);


#endif // RANDOM_NUMBER_GENERATOR_CUH_