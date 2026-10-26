// Command-line conventions shared by the owe tool and the viewer.
#pragma once

#include <stdexcept>
#include <string>
#include <utility>

namespace owe {

// "WxH" as (width, height).
inline std::pair<int, int> parseResolution(const std::string& s) {
    const auto x = s.find('x');
    if (x == std::string::npos || x == 0 || x + 1 == s.size())
        throw std::runtime_error("a resolution is WxH, e.g. 640x480, not '" + s + "'");
    return {std::stoi(s.substr(0, x)), std::stoi(s.substr(x + 1))};
}

// "scene.owe" or "scene.owe:Detector" as (file, detector); a ':' before a '/' belongs to the path.
inline std::pair<std::string, std::string> splitSceneSpec(const std::string& spec) {
    const auto c = spec.rfind(':');
    if (c == std::string::npos || spec.find('/', c) != std::string::npos) return {spec, ""};
    return {spec.substr(0, c), spec.substr(c + 1)};
}

}  // namespace owe
