// owe view — an interactive window over the scenes: choose a view (scene × detector), watch the
// estimate refine pass by pass on any backend (CPU reference, GPU), change display and scene
// values, probe pixels, or tour every view in turn. Orbit, pan and fly a virtual eye through
// the scene without moving the physical sensors or altering the saved detector positions.
#pragma once

#include <string>
#include <vector>

namespace owe {

struct ViewerOptions {
    std::vector<std::string> paths;  // scene files and/or directories of scenes (default: scenes/)
    std::string view;                // initial view: "file.owe" or "file.owe:Detector"
    std::string backend;             // a backend name, or empty (the fastest available: GPU when it runs)
    int device = -1;
    double scale = 0;                // resolution relative to the detector's own (0 = automatic)
    std::vector<std::string> edits;  // Block.key=value edits applied to the initial scene
    bool tour = false;               // cycle through every view
    double tourSeconds = 10;
    std::string screenshot;          // write the window to this PNG after `after` seconds, then exit
    double after = 5;
    int width = 1600, height = 960;
};

// False (with the reason) when the build has no viewer (configure with -DOWE_VIEWER=ON).
bool viewerAvailable(std::string* why = nullptr);
int runViewer(const ViewerOptions& options);

}  // namespace owe
