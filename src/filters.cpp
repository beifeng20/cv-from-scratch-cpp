#include "cvfs/filters.hpp"

#include <algorithm>
#include <cmath>

namespace cvfs {

ImageF to_gray(const ImageU8& img) {
    ImageF out(img.width, img.height, 1);
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x)
            out.at(x, y) = img.channels == 1
                               ? img.at(x, y)
                               : 0.299f * img.at(x, y, 0) + 0.587f * img.at(x, y, 1) + 0.114f * img.at(x, y, 2);
    return out;
}

std::vector<float> gaussian_kernel(float sigma) {
    const int r = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
    std::vector<float> k(2 * r + 1);
    float sum = 0.0f;
    for (int i = -r; i <= r; ++i) {
        k[i + r] = std::exp(-(i * i) / (2.0f * sigma * sigma));
        sum += k[i + r];
    }
    for (float& v : k) v /= sum;
    return k;
}

ImageF gaussian_blur(const ImageF& img, float sigma) {
    if (sigma <= 0.0f) return img;
    const std::vector<float> k = gaussian_kernel(sigma);
    const int r = static_cast<int>(k.size() / 2);
    ImageF tmp(img.width, img.height, 1), out(img.width, img.height, 1);
    // horizontal pass
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x) {
            float s = 0.0f;
            for (int i = -r; i <= r; ++i) s += k[i + r] * img.at(std::clamp(x + i, 0, img.width - 1), y);
            tmp.at(x, y) = s;
        }
    // vertical pass
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x) {
            float s = 0.0f;
            for (int i = -r; i <= r; ++i) s += k[i + r] * tmp.at(x, std::clamp(y + i, 0, img.height - 1));
            out.at(x, y) = s;
        }
    return out;
}

Gradients sobel(const ImageF& img) {
    const int w = img.width, h = img.height;
    Gradients g{ImageF(w, h), ImageF(w, h), ImageF(w, h)};
    auto px = [&](int x, int y) { return img.at(std::clamp(x, 0, w - 1), std::clamp(y, 0, h - 1)); };
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            // Sobel = derivative [-1 0 1] times smoothing [1 2 1] in the other direction
            const float gx = (px(x + 1, y - 1) + 2 * px(x + 1, y) + px(x + 1, y + 1)) -
                             (px(x - 1, y - 1) + 2 * px(x - 1, y) + px(x - 1, y + 1));
            const float gy = (px(x - 1, y + 1) + 2 * px(x, y + 1) + px(x + 1, y + 1)) -
                             (px(x - 1, y - 1) + 2 * px(x, y - 1) + px(x + 1, y - 1));
            g.gx.at(x, y) = gx;
            g.gy.at(x, y) = gy;
            g.magnitude.at(x, y) = std::sqrt(gx * gx + gy * gy);
        }
    return g;
}

ImageU8 to_u8(const ImageF& img, float scale) {
    ImageU8 out(img.width, img.height, img.channels);
    for (std::size_t i = 0; i < img.data.size(); ++i)
        out.data[i] = static_cast<std::uint8_t>(std::clamp(std::lround(img.data[i] * scale), 0L, 255L));
    return out;
}

}  // namespace cvfs
