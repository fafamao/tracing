#ifndef RENDER_CUH_
#define RENDER_CUH_

__global__ render_device(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable **world);

#endif // RENDER_CUH_