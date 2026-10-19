#include "cvfs/hough.hpp"

#include <algorithm>
#include <cmath>

namespace cvfs {
namespace {
constexpr float kPi = 3.14159265358979f;
}

Accumulator hough_accumulate(const ImageU8& edges, const HoughParams& p) {
    Accumulator acc;
    acc.rho_max = std::hypot(static_cast<float>(edges.width), static_cast<float>(edges.height));
    acc.n_theta = p.theta_bins;
    acc.n_rho = static_cast<int>(std::ceil(2.0f * acc.rho_max / p.rho_step)) + 1;  // rho in [-max, max]
    acc.votes.assign(static_cast<std::size_t>(acc.n_rho) * acc.n_theta, 0);

    // cos/sin lookup tables: the inner loop runs (edge pixels x theta bins) times
    std::vector<float> cos_t(acc.n_theta), sin_t(acc.n_theta);
    for (int t = 0; t < acc.n_theta; ++t) {
        const float theta = kPi * t / acc.n_theta;
        cos_t[t] = std::cos(theta);
        sin_t[t] = std::sin(theta);
    }
    for (int y = 0; y < edges.height; ++y)
        for (int x = 0; x < edges.width; ++x) {
            if (!edges.at(x, y)) continue;
            for (int t = 0; t < acc.n_theta; ++t) {
                const float rho = x * cos_t[t] + y * sin_t[t];
                const int r = static_cast<int>(std::lround((rho + acc.rho_max) / p.rho_step));
                ++acc.votes[static_cast<std::size_t>(r) * acc.n_theta + t];
            }
        }
    return acc;
}

std::vector<Line> hough_peaks(const Accumulator& acc, const HoughParams& p) {
    std::vector<Line> lines;
    const int R = p.peak_radius;
    for (int r = 0; r < acc.n_rho; ++r)
        for (int t = 0; t < acc.n_theta; ++t) {
            const int v = acc.at(r, t);
            if (v < p.threshold) continue;
            bool is_max = true;
            for (int dr = -R; dr <= R && is_max; ++dr)
                for (int dt = -R; dt <= R; ++dt) {
                    const int rr = r + dr;
                    if (rr < 0 || rr >= acc.n_rho || (dr == 0 && dt == 0)) continue;
                    // theta wraps around: (rho, theta + pi) is the same line as (-rho, theta)
                    int tt = t + dt, rw = rr;
                    if (tt < 0) { tt += acc.n_theta; rw = acc.n_rho - 1 - rr; }
                    if (tt >= acc.n_theta) { tt -= acc.n_theta; rw = acc.n_rho - 1 - rr; }
                    const int u = acc.at(rw, tt);
                    // strict ">" for cells before us, ">=" after: one winner on plateaus
                    const bool before = (rw < r) || (rw == r && tt < t);
                    if (u > v || (u == v && before)) { is_max = false; break; }
                }
            if (is_max) lines.push_back({r * p.rho_step - acc.rho_max, kPi * t / acc.n_theta, v});
        }
    std::sort(lines.begin(), lines.end(), [](const Line& a, const Line& b) { return a.votes > b.votes; });
    if (static_cast<int>(lines.size()) > p.max_lines) lines.resize(p.max_lines);
    return lines;
}

std::vector<Line> hough_lines(const ImageU8& edges, const HoughParams& p) {
    return hough_peaks(hough_accumulate(edges, p), p);
}

void draw_line(ImageU8& rgb, const Line& line, std::array<std::uint8_t, 3> color, int thickness) {
    const float c = std::cos(line.theta), s = std::sin(line.theta);
    const float x0 = c * line.rho, y0 = s * line.rho;  // closest point of the line to the origin
    const float len = std::hypot(static_cast<float>(rgb.width), static_cast<float>(rgb.height));
    // walk along the line direction (-s, c) in 0.5-pixel steps
    for (float k = -len; k <= len; k += 0.5f) {
        const int x = static_cast<int>(std::lround(x0 - k * s));
        const int y = static_cast<int>(std::lround(y0 + k * c));
        for (int dy = -thickness / 2; dy <= thickness / 2; ++dy)
            for (int dx = -thickness / 2; dx <= thickness / 2; ++dx)
                if (rgb.inside(x + dx, y + dy))
                    for (int ch = 0; ch < 3; ++ch) rgb.at(x + dx, y + dy, ch) = color[ch];
    }
}

ImageU8 accumulator_image(const Accumulator& acc) {
    ImageU8 img(acc.n_theta, acc.n_rho, 1);
    const int vmax = std::max(1, *std::max_element(acc.votes.begin(), acc.votes.end()));
    const float denom = std::log1p(static_cast<float>(vmax));
    for (std::size_t i = 0; i < acc.votes.size(); ++i)
        img.data[i] = static_cast<std::uint8_t>(255.0f * std::log1p(static_cast<float>(acc.votes[i])) / denom);
    return img;
}

}  // namespace cvfs
