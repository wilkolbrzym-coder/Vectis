// ===========================================================================
// Vectis benchmark harness
// ===========================================================================
//
// Small on purpose, and built around two refusals:
//
//   * It refuses to report a single number.  Every kernel is timed many times
//     and the *minimum* is what gets shown, because the minimum is the run
//     least disturbed by the scheduler and by turbo transitions - the median
//     on a shared CI runner is mostly a measurement of the neighbours.
//
//   * It refuses to report a time without a size.  Every kernel here is either
//     compute-bound or bandwidth-bound depending on how much data it touches,
//     and a speedup number with no working-set size attached is not a result.
//
// The workload sizes are chosen to fall into distinct cache levels so the
// crossover is visible rather than argued about.
//
// ===========================================================================
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#  include <x86intrin.h>
#  define VBENCH_HAS_TSC 1
#endif

namespace vbench {

using clock_type = std::chrono::steady_clock;

/// One measured configuration.
struct result {
    std::string name;
    double ns_min = 0.0;        ///< fastest observed run
    double ns_med = 0.0;        ///< median run, for spread reporting
    double cycles_per_elem = 0.0;
    double bytes_per_sec = 0.0;
    double elems_per_sec = 0.0;
};

struct options {
    int    reps      = 21;
    int    warmup    = 3;
    std::size_t bytes = 0;      ///< bytes moved per invocation, for GB/s
    std::size_t elems = 0;      ///< elements processed, for cycles/elem
};

namespace detail {

/// A sink the optimiser cannot see through, so a computed result is never
/// discarded as dead code.
inline volatile double g_sink = 0.0;

[[nodiscard]] inline double now_ns() noexcept {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            clock_type::now().time_since_epoch()).count());
}

} // namespace detail

/// Time `f` repeatedly.  `f` must return a number; it is accumulated into an
/// opaque sink so nothing is optimised away.
template <class F>
[[nodiscard]] result measure(std::string name, const options& opt, F&& f) {
    for (int i = 0; i < opt.warmup; ++i) {
        detail::g_sink = detail::g_sink + static_cast<double>(f());
    }

    std::vector<double> ns;
    ns.reserve(static_cast<std::size_t>(opt.reps));

#ifdef VBENCH_HAS_TSC
    std::uint64_t cycles = 0;
    {
        const auto c0 = __rdtsc();
        for (int i = 0; i < opt.reps; ++i) {
            const double t0 = detail::now_ns();
            detail::g_sink = detail::g_sink + static_cast<double>(f());
            ns.push_back(detail::now_ns() - t0);
        }
        cycles = __rdtsc() - c0;
    }
#else
    for (int i = 0; i < opt.reps; ++i) {
        const double t0 = detail::now_ns();
        detail::g_sink = detail::g_sink + static_cast<double>(f());
        ns.push_back(detail::now_ns() - t0);
    }
#endif

    result r;
    r.name = std::move(name);
    std::sort(ns.begin(), ns.end());
    r.ns_min = ns.front();
    r.ns_med = ns[ns.size() / 2];

    if (opt.elems != 0) {
        r.elems_per_sec = static_cast<double>(opt.elems) / (r.ns_min * 1e-9);
#ifdef VBENCH_HAS_TSC
        r.cycles_per_elem =
            static_cast<double>(cycles) /
            (static_cast<double>(opt.reps) * static_cast<double>(opt.elems));
#endif
    }
    if (opt.bytes != 0) {
        r.bytes_per_sec = static_cast<double>(opt.bytes) / (r.ns_min * 1e-9);
    }
    return r;
}

// ------------------------------------------------------------------ printing

inline void header(const char* title) {
    std::printf("\n=== %s ===\n", title);
}

inline void rule() {
    std::printf("  %-30s %10s %10s %10s %8s\n",
                "kernel", "ns", "cyc/elem", "GB/s", "speedup");
    std::printf("  %-30s %10s %10s %10s %8s\n",
                "------", "--", "--------", "----", "-------");
}

inline void print(const result& r, double baseline_ns) {
    char cyc[32] = "-";
    char gbs[32] = "-";
    if (r.cycles_per_elem > 0.0) {
        std::snprintf(cyc, sizeof cyc, "%.3f", r.cycles_per_elem);
    }
    if (r.bytes_per_sec > 0.0) {
        std::snprintf(gbs, sizeof gbs, "%.1f", r.bytes_per_sec / 1e9);
    }
    const double speedup = (baseline_ns > 0.0 && r.ns_min > 0.0)
                               ? baseline_ns / r.ns_min
                               : 1.0;
    std::printf("  %-30s %10.1f %10s %10s %7.2fx\n",
                r.name.c_str(), r.ns_min, cyc, gbs, speedup);
}

/// Convenience: run a list of alternatives, treat the first as the baseline,
/// and print the table.
template <class Make>
void compare(const char* title, const options& opt,
             std::initializer_list<const char*> names, Make&& make) {
    header(title);
    rule();
    double baseline = 0.0;
    bool first = true;
    for (const char* n : names) {
        const result r = measure(n, opt, make(n));
        if (first) {
            baseline = r.ns_min;
            first = false;
        }
        print(r, baseline);
    }
}

inline void note(const char* text) {
    std::printf("  note: %s\n", text);
}

inline void total_sink() {
    std::printf("\n  (sink %g)\n", static_cast<double>(detail::g_sink));
}

} // namespace vbench
