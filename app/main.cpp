// cvfs_demo: runs the full pipeline on a PPM/PGM image and writes every intermediate stage.
//
//   cvfs_demo <input.ppm> <out_dir> [--sigma 1.4] [--low 50] [--high 150]
//             [--votes 60] [--lines 1] [--roi 0.6] [--min-angle 25]
//
// --roi 0.6       ignore edges above 60% of the image height (sky, trees)
// --min-angle 25  drop lines within 25 degrees of horizontal (lane lines are steep)
// --lines 1       lines to keep on each side (left / right)
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "cvfs/canny.hpp"
#include "cvfs/hough.hpp"

using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <input.ppm> <out_dir> [options]\n", argv[0]);
        return 2;
    }
    const std::string in = argv[1], out = argv[2];
    cvfs::CannyParams cp;
    cvfs::HoughParams hp;
    hp.threshold = 60;
    hp.max_lines = 1;
    float roi = 0.6f, min_angle_deg = 25.0f;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        const float v = std::strtof(argv[i + 1], nullptr);
        if (k == "--sigma") cp.sigma = v;
        else if (k == "--low") cp.low = v;
        else if (k == "--high") cp.high = v;
        else if (k == "--votes") hp.threshold = static_cast<int>(v);
        else if (k == "--lines") hp.max_lines = static_cast<int>(v);
        else if (k == "--roi") roi = v;
        else if (k == "--min-angle") min_angle_deg = v;
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
    }

    try {
        auto t = Clock::now();
        cvfs::ImageU8 img = cvfs::read_pnm(in);
        const double t_read = ms_since(t);

        t = Clock::now();
        const cvfs::ImageF gray = cvfs::to_gray(img);
        const cvfs::ImageF blur = cvfs::gaussian_blur(gray, cp.sigma);
        const double t_blur = ms_since(t);

        t = Clock::now();
        const cvfs::Gradients g = cvfs::sobel(blur);
        const double t_sobel = ms_since(t);

        t = Clock::now();
        const cvfs::ImageF nms = cvfs::non_max_suppression(g);
        cvfs::ImageU8 edges = cvfs::hysteresis(nms, cp.low, cp.high);
        const double t_canny = ms_since(t);

        // region of interest: keep only the lower part of the image
        for (int y = 0; y < static_cast<int>(roi * edges.height); ++y)
            for (int x = 0; x < edges.width; ++x) edges.at(x, y) = 0;

        t = Clock::now();
        const cvfs::Accumulator acc = cvfs::hough_accumulate(edges, hp);
        const int keep = hp.max_lines;  // per side
        hp.max_lines = 200;             // candidate peaks, filtered by angle and side below
        std::vector<cvfs::Line> lines = cvfs::hough_peaks(acc, hp);
        const double t_hough = ms_since(t);

        // theta is the angle of the line NORMAL, so the line's angle to the horizontal is |theta - 90 deg|
        const float min_angle = min_angle_deg * 3.14159265f / 180.0f;
        cvfs::ImageU8 vis = img.channels == 3 ? img : cvfs::ImageU8(img.width, img.height, 3);
        if (img.channels == 1)
            for (int y = 0; y < img.height; ++y)
                for (int x = 0; x < img.width; ++x)
                    for (int c = 0; c < 3; ++c) vis.at(x, y, c) = img.at(x, y);
        // One long lane marking produces many near-duplicate peaks, so keep the strongest
        // line per side: theta < 90 deg leans like a left lane line, theta > 90 deg like a right one.
        int per_side[2] = {0, 0};
        for (const auto& l : lines) {
            if (std::fabs(l.theta - 3.14159265f / 2) < min_angle) continue;
            const int side = l.theta < 3.14159265f / 2 ? 0 : 1;
            if (per_side[side] >= keep) continue;
            ++per_side[side];
            cvfs::draw_line(vis, l, side == 0 ? std::array<std::uint8_t, 3>{40, 220, 40}
                                              : std::array<std::uint8_t, 3>{255, 40, 40}, 3);
            std::printf("%s line rho=%7.1f theta=%5.1f deg votes=%d\n", side == 0 ? "left " : "right", l.rho,
                        l.theta * 180.0f / 3.14159265f, l.votes);
        }

        cvfs::write_pnm(out + "/1_gray.pgm", cvfs::to_u8(gray));
        cvfs::write_pnm(out + "/2_blur.pgm", cvfs::to_u8(blur));
        cvfs::write_pnm(out + "/3_gradient.pgm", cvfs::to_u8(g.magnitude, 0.25f));
        cvfs::write_pnm(out + "/4_edges.pgm", edges);
        cvfs::write_pnm(out + "/5_hough_space.pgm", cvfs::accumulator_image(acc));
        cvfs::write_pnm(out + "/6_lines.ppm", vis);

        std::printf("\n%dx%d image, timings (ms):\n", img.width, img.height);
        std::printf("  read        %7.2f\n  gray+blur   %7.2f\n  sobel       %7.2f\n", t_read, t_blur, t_sobel);
        std::printf("  nms+hyst    %7.2f\n  hough       %7.2f\n", t_canny, t_hough);
        std::printf("  total       %7.2f\n", t_blur + t_sobel + t_canny + t_hough);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
