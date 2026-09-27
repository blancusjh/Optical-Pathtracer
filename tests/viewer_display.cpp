// Render the real viewer, then inspect both halves of its viewport. Merely checking
// that a PNG exists missed a software-renderer bug that hid one textured triangle.
#include "../apps/viewer.hpp"
#include <zlib.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

int main() {
    namespace fs = std::filesystem;
    try {
        fs::path dir = fs::current_path() / "viewer-display";
        fs::create_directories(dir);
        std::ofstream(dir / "uniform.owe") <<
            "world { sky = uniform(1) }\n"
            "observer Eye { position=(0,0,0) look_at=(0,1,0) resolution=(800,500) }\n"
            "render { detector=Eye spp=8 }\n";
        owe::ViewerOptions opt;
        opt.paths = {(dir / "uniform.owe").string()}; opt.backend = "cpu"; opt.scale = 1;
        opt.width = 1000; opt.height = 700; opt.after = 0.2;
        opt.screenshot = (dir / "viewport.png").string();
        if (owe::runViewer(opt) != 0) throw std::runtime_error("viewer failed");
        std::ifstream in(opt.screenshot, std::ios::binary);
        std::vector<unsigned char> png{std::istreambuf_iterator<char>(in), {}};
        auto u32 = [&](size_t p) {
            return (unsigned(png.at(p)) << 24) | (unsigned(png.at(p+1)) << 16) |
                   (unsigned(png.at(p+2)) << 8) | unsigned(png.at(p+3));
        };
        unsigned width = u32(16), height = u32(20);
        std::vector<unsigned char> compressed;
        for (size_t p = 8; p + 12 <= png.size();) {
            size_t n = u32(p);
            if (p + 12 + n > png.size()) throw std::runtime_error("invalid PNG chunk");
            if (std::string(reinterpret_cast<char*>(png.data() + p + 4), 4) == "IDAT")
                compressed.insert(compressed.end(), png.begin() + p + 8, png.begin() + p + 8 + n);
            p += n + 12;
        }
        // The engine's PNG writer emits RGB8, with filter 0 on each scanline.
        std::vector<unsigned char> raw(size_t(height) * (3 * width + 1));
        uLongf length = raw.size();
        if (uncompress(raw.data(), &length, compressed.data(), compressed.size()) != Z_OK || length != raw.size())
            throw std::runtime_error("could not decode viewer PNG");
        for (double fy : {0.35, 0.65}) {
            size_t row = size_t(height * fy) * (3 * width + 1);
            if (raw[row] != 0) throw std::runtime_error("unexpected PNG filter");
            size_t p = row + 1 + 3 * size_t(width * 0.8);
            for (int c = 0; c < 3; ++c) if (raw[p+c] < 70 || raw[p+c] > 240)
                throw std::runtime_error("viewport is missing or corrupt on one side of its diagonal");
        }
        std::cout << "Both halves of the actual viewer image are visible.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
