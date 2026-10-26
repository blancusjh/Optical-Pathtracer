// Length units of the scene language and lens files.
#pragma once

#include <string>

namespace owe {

// Metres per named length unit (m, cm, mm, um, nm, km, in); 0 for an unknown name.
inline double lengthUnitMetres(const std::string& name) {
    if (name == "m") return 1;
    if (name == "cm") return 1e-2;
    if (name == "mm") return 1e-3;
    if (name == "um") return 1e-6;
    if (name == "nm") return 1e-9;
    if (name == "km") return 1e3;
    if (name == "in") return 0.0254;
    return 0;
}

}  // namespace owe
