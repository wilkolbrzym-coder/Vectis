// ===========================================================================
// Vectis test framework - deliberately tiny
// ===========================================================================
//
// No gtest, no download, no build step.  What it does have is the one thing a
// SIMD library actually needs: ULP-distance comparison, because "close enough"
// is not a useful question when the whole point is comparing a vector kernel
// against a scalar oracle that rounds differently.
//
//   VECTIS_TEST(name) { ... }     register a test
//   CHECK(cond)                   record a failure, keep going
//   CHECK_EQ(a, b)                record a failure with both values
//   CHECK_NEAR(a, b, eps)         absolute tolerance
//   CHECK_ULP(a, b, max_ulp)      floating-point distance in ULPs
//   REQUIRE(cond)                 record a failure and stop this test
//
// ===========================================================================
#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string_view>
#include <vector>

namespace vtest {

// ------------------------------------------------------------------ registry

using test_fn = void (*)();

struct test_case {
    const char* name;
    const char* file;
    int         line;
    test_fn     fn;
};

inline std::vector<test_case>& registry() {
    static std::vector<test_case> r;
    return r;
}
inline int& failure_count() { static int n = 0; return n; }
inline int& check_count()   { static int n = 0; return n; }
inline const char*& current_test() { static const char* n = ""; return n; }

struct registrar {
    registrar(const char* name, const char* file, int line, test_fn fn) {
        registry().push_back({name, file, line, fn});
    }
};

// ----------------------------------------------------------------- reporting

inline void report_failure(const char* what, const char* file, int line) {
    ++failure_count();
    if (failure_count() <= 50) {
        std::fprintf(stderr, "  FAIL %s:%d  [%s]\n    %s\n",
                     file, line, current_test(), what);
    } else if (failure_count() == 51) {
        std::fprintf(stderr, "  ... further failures suppressed\n");
    }
}

template <class T>
inline void format_value(char* out, std::size_t n, const T& v) {
    if constexpr (std::is_floating_point_v<T>) {
        std::snprintf(out, n, "%.9g", static_cast<double>(v));
    } else if constexpr (std::is_integral_v<T>) {
        if constexpr (std::is_signed_v<T>) {
            std::snprintf(out, n, "%lld", static_cast<long long>(v));
        } else {
            std::snprintf(out, n, "%llu", static_cast<unsigned long long>(v));
        }
    } else if constexpr (std::is_same_v<T, bool>) {
        std::snprintf(out, n, "%s", v ? "true" : "false");
    } else {
        std::snprintf(out, n, "<value>");
    }
}

inline bool check_bool(bool ok, const char* expr, const char* file, int line) {
    ++check_count();
    if (!ok) report_failure(expr, file, line);
    return ok;
}

template <class A, class B>
inline bool check_eq(const A& a, const B& b, const char* ea, const char* eb,
                     const char* file, int line) {
    ++check_count();
    if (a == b) return true;
    char ba[96], bb[96];
    format_value(ba, sizeof ba, a);
    format_value(bb, sizeof bb, b);
    char msg[640];
    std::snprintf(msg, sizeof msg, "%s == %s  (got %s vs %s)", ea, eb, ba, bb);
    report_failure(msg, file, line);
    return false;
}

template <class T, class U>
inline bool check_near(T a, U b, double eps, const char* ea, const char* eb,
                       const char* file, int line) {
    ++check_count();
    const double d = std::fabs(static_cast<double>(a) - static_cast<double>(b));
    if (d <= eps) return true;
    char msg[640];
    std::snprintf(msg, sizeof msg, "|%s - %s| <= %g  (diff %.9g)", ea, eb, eps, d);
    report_failure(msg, file, line);
    return false;
}

// ------------------------------------------------------------- ULP distance
//
// The bit patterns of IEEE-754 floats are monotone in value once the sign bit
// is folded away, so the integer distance between the folded patterns is
// exactly the number of representable values between them.  That is the only
// comparison that means anything when two code paths round differently.

[[nodiscard]] inline std::uint32_t fold(float x) noexcept {
    const std::uint32_t b = std::bit_cast<std::uint32_t>(x);
    return (b & 0x8000'0000u) ? ~b : (b | 0x8000'0000u);
}
[[nodiscard]] inline std::uint64_t fold(double x) noexcept {
    const std::uint64_t b = std::bit_cast<std::uint64_t>(x);
    return (b & 0x8000'0000'0000'0000ull) ? ~b : (b | 0x8000'0000'0000'0000ull);
}

[[nodiscard]] inline std::uint64_t ulp_distance(float a, float b) noexcept {
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b) ? 0 : UINT64_MAX;
    if (a == b) return 0;                       // covers +0 vs -0
    const auto fa = fold(a), fb = fold(b);
    return fa > fb ? fa - fb : fb - fa;
}
[[nodiscard]] inline std::uint64_t ulp_distance(double a, double b) noexcept {
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b) ? 0 : UINT64_MAX;
    if (a == b) return 0;
    const auto fa = fold(a), fb = fold(b);
    return fa > fb ? fa - fb : fb - fa;
}

template <class T>
inline bool check_ulp(T a, T b, std::uint64_t max_ulp, const char* ea,
                      const char* eb, const char* file, int line) {
    ++check_count();
    const std::uint64_t d = ulp_distance(a, b);
    if (d <= max_ulp) return true;
    char ba[96], bb[96];
    format_value(ba, sizeof ba, a);
    format_value(bb, sizeof bb, b);
    char msg[640];
    std::snprintf(msg, sizeof msg,
                  "%s ~= %s within %llu ulp  (got %s vs %s, %llu ulp)",
                  ea, eb, static_cast<unsigned long long>(max_ulp), ba, bb,
                  static_cast<unsigned long long>(d));
    report_failure(msg, file, line);
    return false;
}

} // namespace vtest

// ------------------------------------------------------------------- macros

#define VECTIS_TEST(name)                                                      \
    static void vtest_fn_##name();                                             \
    static ::vtest::registrar vtest_reg_##name(#name, __FILE__, __LINE__,      \
                                               &vtest_fn_##name);              \
    static void vtest_fn_##name()

#define CHECK(cond)                                                            \
    ::vtest::check_bool(static_cast<bool>(cond), #cond, __FILE__, __LINE__)

#define CHECK_EQ(a, b)                                                         \
    ::vtest::check_eq((a), (b), #a, #b, __FILE__, __LINE__)

#define CHECK_NEAR(a, b, eps)                                                  \
    ::vtest::check_near((a), (b), (eps), #a, #b, __FILE__, __LINE__)

#define CHECK_ULP(a, b, max_ulp)                                               \
    ::vtest::check_ulp((a), (b), (max_ulp), #a, #b, __FILE__, __LINE__)

#define REQUIRE(cond)                                                          \
    do {                                                                       \
        if (!::vtest::check_bool(static_cast<bool>(cond), #cond, __FILE__,      \
                                 __LINE__)) {                                   \
            return;                                                            \
        }                                                                      \
    } while (0)

// ------------------------------------------------------------------ main

// Test files link against this: one definition, in test_main.cpp.
int vtest_main(int argc, char** argv);
