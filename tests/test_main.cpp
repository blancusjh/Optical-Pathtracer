#include <chrono>
#include <cstring>
#include <cstdlib>
#include "owe/backends/registry.hpp"

#include "check.hpp"

std::vector<TestCase>& testRegistry() {
    static std::vector<TestCase> r;
    return r;
}
int g_checkFailures = 0;

int main(int argc, char** argv) {
    if (std::getenv("OWE_REQUIRE_GPU")) {
        std::string reason;
        if (!owe::backend("gpu").available(&reason)) {
            std::fprintf(stderr, "GPU tests required but unavailable: %s\n", reason.c_str());
            return 1;
        }
    }
    // Arguments: substrings; a test runs when its name contains any of them (all tests without).
    int failedTests = 0, ran = 0;
    for (auto& t : testRegistry()) {
        bool selected = argc < 2;
        for (int i = 1; i < argc && !selected; ++i) selected = std::strstr(t.name, argv[i]) != nullptr;
        if (!selected) continue;
        int before = g_checkFailures;
        auto t0 = std::chrono::steady_clock::now();
        std::fprintf(stderr, "[ RUN  ] %s\n", t.name);
        try {
            t.fn();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "    exception: %s\n", e.what());
            ++g_checkFailures;
        }
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        bool ok = g_checkFailures == before;
        if (!ok) ++failedTests;
        ++ran;
        std::fprintf(stderr, "[ %s ] %s (%.2f s)\n", ok ? " OK " : "FAIL", t.name, s);
    }
    std::fprintf(stderr, "\n%d/%d tests passed\n", ran - failedTests, ran);
    if (ran == 0) std::fprintf(stderr, "No tests matched the requested filters\n");
    return failedTests || ran == 0 ? 1 : 0;
}
