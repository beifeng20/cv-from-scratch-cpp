#pragma once
// Canny edge detector: blur -> Sobel -> non-maximum suppression -> double threshold + hysteresis.
#include "cvfs/filters.hpp"

namespace cvfs {

struct CannyParams {
    float sigma = 1.4f;   // Gaussian blur; 0 disables it
    float low = 50.0f;    // weak-edge threshold on gradient magnitude
    float high = 150.0f;  // strong-edge threshold
};

// Keeps a pixel only if its magnitude is a local maximum along the gradient direction
// (quantized to 0/45/90/135 degrees). Result: edges one pixel wide.
ImageF non_max_suppression(const Gradients& g);

// Pixels >= high are edges; pixels in [low, high) become edges only if they are
// 8-connected to a strong edge (iterative flood fill, no recursion).
ImageU8 hysteresis(const ImageF& nms, float low, float high);

// Full pipeline on a grayscale float image. Output: 255 = edge, 0 = background.
ImageU8 canny(const ImageF& gray, const CannyParams& p = {});

}  // namespace cvfs
