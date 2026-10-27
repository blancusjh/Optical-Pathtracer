// Deterministic random numbers of the CPU reference. The warps that turn them into samples are
// shared with the GPU kernels (owe/core/shared/warps.hpp).
#pragma once

#include "owe/core/math.hpp"

namespace owe {

// PCG32 (O'Neill). Streams are fully determined by (seed, stream id), which
// makes every render reproducible independent of thread scheduling.
class Rng {
public:
    Rng() { seed(0, 0); }
    Rng(uint64_t seedValue, uint64_t stream) { seed(seedValue, stream); }
    void seed(uint64_t seedValue, uint64_t stream) {
        state_ = 0;
        inc_ = (stream << 1u) | 1u;
        nextU32();
        state_ += seedValue;
        nextU32();
    }
    uint32_t nextU32() { return pcg32Next(state_, inc_); }
    // Uniform in [0,1) with 53 bits.
    double uniform() {
        uint64_t a = nextU32() >> 5, b = nextU32() >> 6;
        return (double(a) * 67108864.0 + double(b)) * (1.0 / 9007199254740992.0);
    }

private:
    uint64_t state_, inc_;
};

// Several uniforms for one sampling call, drawn in a fixed order. C++ leaves the evaluation
// order of `f(rng.uniform(), rng.uniform())` unspecified (GCC draws the last argument first,
// Clang the first), which made renders depend on the compiler. The order is now explicit and is
// the one GCC used, so earlier renders reproduce; the GPU kernels draw in the same order.
struct Draw2 {
    double u1, u2;
    explicit Draw2(Rng& r) {
        u2 = r.uniform();
        u1 = r.uniform();
    }
};
struct Draw3 {
    double u1, u2, u3;
    explicit Draw3(Rng& r) {
        u3 = r.uniform();
        u2 = r.uniform();
        u1 = r.uniform();
    }
};

}  // namespace owe
