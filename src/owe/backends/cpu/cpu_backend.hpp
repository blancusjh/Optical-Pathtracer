// The reference backend: the reference transport (owe/transport, IEEE-754 double throughout) run
// on host threads. Renders are bitwise reproducible for a seed, independent of the thread count and
// of the compiler, and implement every integrator (path, light, hybrid). Other backends are
// validated against this one.
#pragma once

#include "owe/render/backend.hpp"

namespace owe {

const Backend& cpuBackend();

}  // namespace owe
