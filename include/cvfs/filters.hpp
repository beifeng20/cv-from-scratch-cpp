#pragma once
// Grayscale conversion, Gaussian blur and Sobel gradients.
#include <vector>

#include "cvfs/image.hpp"

namespace cvfs {

// RGB -> luminance (ITU-R BT.601 weights), output in [0, 255] as float.
ImageF to_gray(const ImageU8& img);

// Normalized 1-D Gaussian kernel, radius = ceil(3 * sigma).
std::vector<float> gaussian_kernel(float sigma);

// Separable Gaussian blur (rows, then columns): O(k) per pixel instead of O(k^2).
// Borders are handled by clamping coordinates (replicate the edge pixel).
ImageF gaussian_blur(const ImageF& img, float sigma);

struct Gradients {
    ImageF gx;         // d/dx, positive when brightness grows to the right
    ImageF gy;         // d/dy, positive when brightness grows downwards
    ImageF magnitude;  // sqrt(gx^2 + gy^2)
};

// 3x3 Sobel operator.
Gradients sobel(const ImageF& img);

// Clamp to [0, 255] and round; `scale` multiplies values first.
ImageU8 to_u8(const ImageF& img, float scale = 1.0f);

}  // namespace cvfs
