// Image output, identical for every backend. The raw estimate (PFM, linear sRGB) is never touched
// by display processing; the display image (PNG, and the viewer's texture) applies exposure,
// optional white balance, gamut mapping, a view transform (tone curve) and sRGB encoding.
#pragma once

#include <string>
#include <vector>

#include "owe/scene/detector.hpp"

namespace owe {

void writePFM(const std::string& path, const Image& img);                 // linear sRGB, raw
Image readPFM(const std::string& path);                                   // as writePFM wrote it
// Display meter gain in stops (meterEV, owe/render/exposure.hpp); never changes the raw estimate.
double autoExposureEV(const Image& img);
// Brings a linear sRGB colour with a negative component into the gamut by reducing its chroma at
// constant Oklab lightness and hue (every monochromatic colour needs this). In-gamut colours are
// unchanged.
void gamutMapLinearSRGB(double rgb[3]);
// The display's view transform, from scene-linear to display-linear sRGB.
//  AgX (Sobotka): a log2 encoding of 16.5 stops (scene values 0.00018 to 16.3), a sigmoid, and a
//    slight inset of the primaries, so the brightest colours go gradually to white, as the core of
//    a caustic, a sun glint or a bright spectrum looks on film and to the eye. Middle grey 0.18
//    shows at 0.215.
//  Standard: linear up to white (1.0) with a short roll-off, and colours beyond white fade to
//    white at their own luminance. Middle grey 0.18 also shows at 0.215.
enum class Tone { AgX, Standard };
Tone toneFromName(const std::string& name);  // "agx" | "standard"; throws otherwise
void applyTone(Tone tone, double rgb[3]);    // linear sRGB (non-negative) → display-linear [0, 1]
// Display rendering: exposure, optional white balance (Bradford adaptation from a Planckian white
// of `whiteKelvin` to D65; 0 = none), gamut mapping (gamutMapLinearSRGB), the view transform,
// sRGB encoding. Never applied to PFM.
void writePNG(const std::string& path, const Image& img, double exposureEV, bool autoExposure, double whiteKelvin = 0,
              Tone tone = Tone::AgX);
// The same display rendering as 8-bit sRGB triples, row by row from the top (as written to PNG).
std::vector<unsigned char> displayRGB8(const Image& img, double exposureEV, bool autoExposure, double whiteKelvin = 0,
                                       Tone tone = Tone::AgX);
// Writes 8-bit RGB triples (row by row from the top) as a PNG.
void writeRGB8PNG(const std::string& path, int width, int height, const std::vector<unsigned char>& rgb);

}  // namespace owe
