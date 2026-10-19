// Unit tests without a framework: each CHECK prints the failing expression and line.
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>

#include "cvfs/canny.hpp"
#include "cvfs/hough.hpp"

static int g_failures = 0;
#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);   \
            ++g_failures;                                                 \
        }                                                                 \
    } while (0)

using namespace cvfs;

// Image with a vertical step: left half dark (20), right half bright (220).
static ImageF vertical_step(int w = 40, int h = 30) {
    ImageF img(w, h, 1, 20.0f);
    for (int y = 0; y < h; ++y)
        for (int x = w / 2; x < w; ++x) img.at(x, y) = 220.0f;
    return img;
}

static void test_gaussian_kernel() {
    const auto k = gaussian_kernel(1.4f);
    CHECK(k.size() == 11);  // radius ceil(3 * 1.4) = 5
    CHECK(std::fabs(std::accumulate(k.begin(), k.end(), 0.0f) - 1.0f) < 1e-5f);
    CHECK(k[5] > k[4] && k[4] > k[3]);         // peak in the middle
    CHECK(std::fabs(k[2] - k[8]) < 1e-7f);     // symmetric
}

static void test_blur_keeps_constant_image() {
    const ImageF flat(16, 16, 1, 100.0f);
    const ImageF b = gaussian_blur(flat, 2.0f);
    for (float v : b.data) CHECK(std::fabs(v - 100.0f) < 1e-3f);
}

static void test_sobel_direction() {
    const Gradients g = sobel(vertical_step());
    // at the step, brightness grows to the right: gx > 0, gy == 0
    CHECK(g.gx.at(20, 15) > 0.0f);
    CHECK(std::fabs(g.gy.at(20, 15)) < 1e-4f);
    CHECK(g.magnitude.at(5, 15) == 0.0f);  // flat area
}

static void test_canny_thin_vertical_edge() {
    const ImageU8 e = canny(vertical_step(), {1.0f, 50.0f, 150.0f});
    for (int y = 3; y < 27; ++y) {
        int count = 0;
        for (int x = 0; x < e.width; ++x) count += e.at(x, y) > 0;
        CHECK(count == 1);  // non-maximum suppression -> exactly one pixel wide
    }
}

static void test_hysteresis_connectivity() {
    ImageF nms(10, 3, 1, 0.0f);
    nms.at(1, 1) = 200;                                // strong
    for (int x = 2; x <= 5; ++x) nms.at(x, 1) = 80;    // weak chain connected to it
    nms.at(8, 1) = 80;                                 // isolated weak pixel
    const ImageU8 out = hysteresis(nms, 50, 150);
    CHECK(out.at(1, 1) == 255);
    CHECK(out.at(5, 1) == 255);
    CHECK(out.at(8, 1) == 0);
}

static void test_hough_finds_known_line() {
    // draw the line x = 30 (vertical): normal angle theta = 0, rho = 30
    ImageU8 edges(80, 60, 1, 0);
    for (int y = 0; y < 60; ++y) edges.at(30, y) = 255;
    // and a diagonal y = x: normal at 135 degrees, rho = 0
    for (int i = 0; i < 60; ++i) edges.at(i, i) = 255;
    HoughParams p;
    p.threshold = 40;
    const auto lines = hough_lines(edges, p);
    CHECK(lines.size() >= 2);
    bool vertical = false, diagonal = false;
    for (const auto& l : lines) {
        const float deg = l.theta * 180.0f / 3.14159265f;
        if (std::fabs(l.rho - 30.0f) <= 1.0f && deg <= 1.0f) vertical = true;
        if (std::fabs(l.rho) <= 1.0f && std::fabs(deg - 135.0f) <= 1.0f) diagonal = true;
    }
    CHECK(vertical);
    CHECK(diagonal);
}

static void test_pnm_roundtrip() {
    ImageU8 img(5, 4, 3);
    for (std::size_t i = 0; i < img.data.size(); ++i) img.data[i] = static_cast<std::uint8_t>(i * 7);
    const std::string path = "roundtrip_test.ppm";
    write_pnm(path, img);
    const ImageU8 back = read_pnm(path);
    std::remove(path.c_str());
    CHECK(back.width == 5 && back.height == 4 && back.channels == 3);
    CHECK(back.data == img.data);
}

int main() {
    test_gaussian_kernel();
    test_blur_keeps_constant_image();
    test_sobel_direction();
    test_canny_thin_vertical_edge();
    test_hysteresis_connectivity();
    test_hough_finds_known_line();
    test_pnm_roundtrip();
    if (g_failures) {
        std::printf("%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all tests passed\n");
    return 0;
}
