#pragma once
// Minimal image container + binary PGM/PPM (P5/P6) I/O. No external dependencies.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cvfs {

template <typename T>
struct Image {
    int width = 0;
    int height = 0;
    int channels = 1;
    std::vector<T> data;  // row-major, channels interleaved

    Image() = default;
    Image(int w, int h, int c = 1, T fill = T{})
        : width(w), height(h), channels(c), data(static_cast<std::size_t>(w) * h * c, fill) {}

    T& at(int x, int y, int c = 0) { return data[index(x, y, c)]; }
    const T& at(int x, int y, int c = 0) const { return data[index(x, y, c)]; }
    bool empty() const { return data.empty(); }
    bool inside(int x, int y) const { return x >= 0 && y >= 0 && x < width && y < height; }

private:
    std::size_t index(int x, int y, int c) const {
        return (static_cast<std::size_t>(y) * width + x) * channels + c;
    }
};

using ImageU8 = Image<std::uint8_t>;
using ImageF = Image<float>;

// Reads P5 (grayscale) or P6 (RGB) with maxval 255. Throws std::runtime_error on bad input.
ImageU8 read_pnm(const std::string& path);
// Writes P5 for 1 channel, P6 for 3 channels.
void write_pnm(const std::string& path, const ImageU8& img);

}  // namespace cvfs
