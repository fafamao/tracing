#ifndef CONSTANTS_H
#define CONSTANTS_H

// Picture size
inline constexpr double PIXEL_FACTOR = 255.99;
inline constexpr double PIXEL_SCALE = 16.0 / 9.0;

// Focal length
inline constexpr double FOCAL_LEN = 1.0;

// TODO: Configure pixel size from static file
inline constexpr int PIXEL_WIDTH = 400;
inline constexpr int PIXEL_HEIGHT = int(double(PIXEL_WIDTH) / PIXEL_SCALE);

// Viewport size
/* inline constexpr double VIEWPORT_WIDTH = 4.8;
inline constexpr double VIEWPORT_HEIGHT = VIEWPORT_WIDTH / (PIXEL_WIDTH / PIXEL_HEIGHT); */
inline constexpr double VIEWPORT_HEIGHT = 2.0;
inline constexpr double VIEWPORT_WIDTH = VIEWPORT_HEIGHT / (PIXEL_WIDTH / PIXEL_HEIGHT);

#endif // CONSTANTS_H
