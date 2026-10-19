#include "cvfs/image.hpp"

#include <cctype>
#include <fstream>
#include <stdexcept>

namespace cvfs {
namespace {

// Reads the next header token, skipping whitespace and '#' comments.
std::string next_token(std::istream& in) {
    std::string tok;
    char c;
    while (in.get(c)) {
        if (c == '#') {
            std::string ignore;
            std::getline(in, ignore);
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            if (!tok.empty()) return tok;
        } else {
            tok += c;
        }
    }
    return tok;
}

}  // namespace

ImageU8 read_pnm(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    const std::string magic = next_token(in);
    if (magic != "P5" && magic != "P6") throw std::runtime_error(path + ": only binary P5/P6 is supported");
    const int w = std::stoi(next_token(in));
    const int h = std::stoi(next_token(in));
    const int maxval = std::stoi(next_token(in));  // the single whitespace after it was consumed
    if (w <= 0 || h <= 0 || maxval != 255) throw std::runtime_error(path + ": unsupported header");

    ImageU8 img(w, h, magic == "P5" ? 1 : 3);
    in.read(reinterpret_cast<char*>(img.data.data()), static_cast<std::streamsize>(img.data.size()));
    if (in.gcount() != static_cast<std::streamsize>(img.data.size()))
        throw std::runtime_error(path + ": truncated pixel data");
    return img;
}

void write_pnm(const std::string& path, const ImageU8& img) {
    if (img.channels != 1 && img.channels != 3) throw std::runtime_error("write_pnm: need 1 or 3 channels");
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);
    out << (img.channels == 1 ? "P5" : "P6") << '\n' << img.width << ' ' << img.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(img.data.data()), static_cast<std::streamsize>(img.data.size()));
}

}  // namespace cvfs
