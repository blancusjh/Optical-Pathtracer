// Statistics every transport implementation reports, whichever backend runs it: paths, segments,
// region inconsistencies and leaks.
#pragma once

#include <cstdint>

#include "owe/core/math.hpp"
#include "owe/scene/world.hpp"

namespace owe {

struct TransportStats {
    uint64_t paths = 0, segments = 0, inconsistencies = 0, leaks = 0;
    // First region inconsistency seen: a ray reached a boundary from a region other than
    // the one it was travelling in (overlapping or unclosed geometry).
    struct Inconsistency {
        uint32_t boundary = kNone, rayRegion = kNone, boundaryRegion = kNone, previous = kNone;
        Vec3 p;
    } first;
    void noteInconsistency(uint32_t boundary, uint32_t rayRegion, uint32_t boundaryRegion, uint32_t previous,
                           const Vec3& p) {
        if (inconsistencies++ == 0) first = {boundary, rayRegion, boundaryRegion, previous, p};
    }
    void add(const TransportStats& o) {
        if (inconsistencies == 0 && o.inconsistencies > 0) first = o.first;
        paths += o.paths; segments += o.segments; inconsistencies += o.inconsistencies; leaks += o.leaks;
    }
};

}  // namespace owe
