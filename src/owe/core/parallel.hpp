// Data parallelism for setup work (loading meshes, building hierarchies). Work is split into fixed
// ranges and combined in order, so results never depend on the thread count or on scheduling.
#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace owe {

inline unsigned hardwareThreads() { return std::max(1u, std::thread::hardware_concurrency()); }

// Calls f(part, begin, end) for `parts` consecutive ranges covering [0, n), in parallel. The first
// exception thrown by any part is rethrown once all parts have finished.
template <class F>
void parallelParts(size_t n, size_t parts, F&& f) {
    parts = std::max<size_t>(1, std::min(parts, n));
    if (parts <= 1) {
        if (n) f(size_t(0), size_t(0), n);
        return;
    }
    std::exception_ptr error;
    std::mutex m;
    auto run = [&](size_t p) {
        try {
            f(p, n * p / parts, n * (p + 1) / parts);
        } catch (...) {
            std::lock_guard<std::mutex> lk(m);
            if (!error) error = std::current_exception();
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(parts - 1);
    for (size_t p = 1; p < parts; ++p) pool.emplace_back(run, p);
    run(0);
    for (auto& t : pool) t.join();
    if (error) std::rethrow_exception(error);
}

// Calls f(begin, end) over [0, n) in ranges of at least `grain` items, in parallel.
template <class F>
void parallelFor(size_t n, size_t grain, F&& f) {
    size_t parts = std::min<size_t>(hardwareThreads(), (n + grain - 1) / std::max<size_t>(1, grain));
    parallelParts(n, parts, [&](size_t, size_t begin, size_t end) { f(begin, end); });
}

}  // namespace owe
