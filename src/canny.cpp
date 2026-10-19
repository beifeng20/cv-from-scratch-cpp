#include "cvfs/canny.hpp"

#include <cmath>
#include <utility>
#include <vector>

namespace cvfs {

ImageF non_max_suppression(const Gradients& g) {
    const ImageF& mag = g.magnitude;
    const int w = mag.width, h = mag.height;
    ImageF out(w, h, 1, 0.0f);
    constexpr float kRadToDeg = 57.29577951f;
    for (int y = 1; y < h - 1; ++y)
        for (int x = 1; x < w - 1; ++x) {
            const float m = mag.at(x, y);
            if (m == 0.0f) continue;
            float angle = std::atan2(g.gy.at(x, y), g.gx.at(x, y)) * kRadToDeg;  // (-180, 180]
            if (angle < 0) angle += 180.0f;                                      // direction mod 180
            // neighbours along the gradient direction (image y axis points down)
            int dx, dy;
            if (angle < 22.5f || angle >= 157.5f) { dx = 1; dy = 0; }   // horizontal gradient
            else if (angle < 67.5f)               { dx = 1; dy = 1; }   // down-right diagonal
            else if (angle < 112.5f)              { dx = 0; dy = 1; }   // vertical gradient
            else                                  { dx = -1; dy = 1; }  // down-left diagonal
            const float a = mag.at(x + dx, y + dy), b = mag.at(x - dx, y - dy);
            // ">" on one side and ">=" on the other keeps exactly one pixel on flat ridges
            if (m > a && m >= b) out.at(x, y) = m;
        }
    return out;
}

ImageU8 hysteresis(const ImageF& nms, float low, float high) {
    const int w = nms.width, h = nms.height;
    ImageU8 out(w, h, 1, 0);
    std::vector<std::pair<int, int>> stack;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            if (nms.at(x, y) >= high) {
                out.at(x, y) = 255;
                stack.emplace_back(x, y);
            }
    // grow strong edges into connected weak pixels (depth-first, explicit stack)
    while (!stack.empty()) {
        const auto [x, y] = stack.back();
        stack.pop_back();
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                const int nx = x + dx, ny = y + dy;
                if (!out.inside(nx, ny) || out.at(nx, ny)) continue;
                if (nms.at(nx, ny) >= low) {
                    out.at(nx, ny) = 255;
                    stack.emplace_back(nx, ny);
                }
            }
    }
    return out;
}

ImageU8 canny(const ImageF& gray, const CannyParams& p) {
    const Gradients g = sobel(gaussian_blur(gray, p.sigma));
    return hysteresis(non_max_suppression(g), p.low, p.high);
}

}  // namespace cvfs
