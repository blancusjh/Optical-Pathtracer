#include <chrono>
#include <cstring>

#include "check.hpp"

std::vector<TestCase>& testRegistry() {
    static std::vector<TestCase> r;
    return r;
}
int g_checkFailures = 0;

int main(int argc, char** argv) {
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
    return failedTests ? 1 : 0;
}
