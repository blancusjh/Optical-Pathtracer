// Stub used when the engine is built without the viewer.
#include <cstdio>

#include "viewer.hpp"

namespace owe {

bool viewerAvailable(std::string* why) {
    if (why) *why = "this build has no viewer (configure with -DOWE_VIEWER=ON; it needs SDL3)";
    return false;
}

int runViewer(const ViewerOptions&) {
    std::string why;
    viewerAvailable(&why);
    std::fprintf(stderr, "owe view: %s\n", why.c_str());
    return 2;
}

}  // namespace owe
