// Minimal self-contained test harness.
#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

struct TestCase {
    const char* name;
    std::function<void()> fn;
};
std::vector<TestCase>& testRegistry();
extern int g_checkFailures;

struct TestRegistrar {
    TestRegistrar(const char* n, std::function<void()> f) { testRegistry().push_back({n, std::move(f)}); }
};

#define TEST(name)                                          \
    static void name();                                     \
    static TestRegistrar registrar_##name(#name, name);     \
    static void name()

#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            std::fprintf(stderr, "    %s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
            ++g_checkFailures;                                                             \
        }                                                                                  \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                              \
    do {                                                                                                   \
        double va_ = (a), vb_ = (b), vt_ = (tol);                                                          \
        if (!(std::abs(va_ - vb_) <= vt_)) {                                                               \
            std::fprintf(stderr, "    %s:%d: %s = %.12g, expected %s = %.12g (tol %.3g, diff %.3g)\n", __FILE__, \
                         __LINE__, #a, va_, #b, vb_, vt_, std::abs(va_ - vb_));                            \
            ++g_checkFailures;                                                                             \
        }                                                                                                  \
    } while (0)

#define CHECK_REL(a, b, rel) CHECK_NEAR(a, b, (rel) * std::abs(double(b)))
