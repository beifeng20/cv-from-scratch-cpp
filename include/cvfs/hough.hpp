#pragma once
// Standard Hough transform for lines: rho = x*cos(theta) + y*sin(theta).
#include <array>
#include <cstdint>
#include <vector>

#include "cvfs/image.hpp"

namespace cvfs {

struct Line {
    float rho;    // signed distance from the image origin, pixels
    float theta;  // angle of the line normal, radians in [0, pi)
    int votes;
};

struct HoughParams {
    float rho_step = 1.0f;  // pixels per accumulator row
    int theta_bins = 180;   // 1 degree per bin by default
    int threshold = 80;     // minimum votes for a line
    int max_lines = 20;
    int peak_radius = 6;    // peaks must be the maximum within this window (suppresses duplicates)
};

struct Accumulator {
    int n_rho = 0;
    int n_theta = 0;
    float rho_max = 0.0f;
    std::vector<int> votes;  // n_rho rows x n_theta columns
    int at(int r, int t) const { return votes[static_cast<std::size_t>(r) * n_theta + t]; }
};

// Every edge pixel (value > 0) votes for all (rho, theta) lines through it.
Accumulator hough_accumulate(const ImageU8& edges, const HoughParams& p = {});

// Local maxima of the accumulator above the threshold, strongest first.
std::vector<Line> hough_peaks(const Accumulator& acc, const HoughParams& p = {});

// Convenience: accumulate + peaks.
std::vector<Line> hough_lines(const ImageU8& edges, const HoughParams& p = {});

// Draws an infinite line (clipped to the image) on a 3-channel image.
void draw_line(ImageU8& rgb, const Line& line, std::array<std::uint8_t, 3> color, int thickness = 2);

// Accumulator rendered as an 8-bit image (log-scaled) for visualization.
ImageU8 accumulator_image(const Accumulator& acc);

}  // namespace cvfs
