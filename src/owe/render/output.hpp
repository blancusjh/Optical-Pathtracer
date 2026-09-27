// Image output, identical for every backend. The raw estimate (PFM, linear sRGB) is never touched
// by display processing; the display image (PNG, and the viewer's texture) applies exposure,
// optional white balance, a highlight roll-off and sRGB encoding.
#pragma once

#include <string>
#include <vector>

#include "owe/scene/detector.hpp"

namespace owe {

void writePFM(const std::string& path, const Image& img);                 // linear sRGB, raw
// Display meter gain in stops; never changes the raw estimate.
double autoExposureEV(const Image& img);
// Display rendering: exposure, optional white balance (Bradford adaptation from a Planckian white
// of `whiteKelvin` to D65; 0 = none), highlight roll-off, sRGB encoding. Never applied to PFM.
void writePNG(const std::string& path, const Image& img, double exposureEV, bool autoExposure, double whiteKelvin = 0);
// The same display rendering as 8-bit sRGB triples, row by row from the top (as written to PNG).
std::vector<unsigned char> displayRGB8(const Image& img, double exposureEV, bool autoExposure, double whiteKelvin = 0);
// Writes 8-bit RGB triples (row by row from the top) as a PNG.
void writeRGB8PNG(const std::string& path, int width, int height, const std::vector<unsigned char>& rgb);

}  // namespace owe
