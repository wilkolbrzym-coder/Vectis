// ===========================================================================
// Vectis - what the vector engine actually buys
// ===========================================================================
//
// The scalar column here is a plain C++ loop written the way one would write it
// without the library.  It is compiled with the same -O2 and the same -march as
// everything else, which means the compiler is free to auto-vectorise it - and
// on a simple loop it will.  That is the honest comparison to make: the library
// is competing against a modern optimising compiler, not against a straw man.
//
// The interesting columns are the ones that differ only in lane count.  Same
// source, same kernel, `f32<8>` versus `f32<32>`: on AVX2 that is one YMM per
// operation versus four, and it shows whether the multi-register fold costs
// anything over the single-register case.
//
// ===========================================================================
#include "bench_harness.hpp"

#include <vectis/core/cpu.hpp>
#include <vectis/simd/reduce.hpp>
#include <vectis/simd/vec.hpp>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

using namespace vectis;

namespace {

void fill(std::vector<float>& v, std::uint32_t seed) {
    std::uint32_t s = seed;
    for (auto& x : v) {
        s = s * 1664525u + 1013904223u;
        x = static_cast<float>(static_cast<double>(s >> 8) / 8388608.0 - 1.0);
    }
}

/// out = a*b + a, with the vector width as a compile-time parameter.
template <class V>
float muladd_vec(const float* a, const float* b, float* out, std::size_t n) {
    std::size_t i = 0;
    for (; i + V::lanes <= n; i += V::lanes) {
        const V va = V::load(a + i);
        const V vb = V::load(b + i);
        va.fma(vb, va).store(out + i);
    }
    for (; i < n; ++i) out[i] = a[i] * b[i] + a[i];
    return out[0];
}

float muladd_scalar(const float* a, const float* b, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i] * b[i] + a[i];
    return out[0];
}

template <class V>
float sum_vec(const float* a, std::size_t n) {
    V acc = V::zero();
    std::size_t i = 0;
    for (; i + V::lanes <= n; i += V::lanes) {
        acc = acc + V::load(a + i);
    }
    float total = reduce_add(acc);
    for (; i < n; ++i) total += a[i];
    return total;
}

float sum_scalar(const float* a, std::size_t n) {
    float total = 0.0f;
    for (std::size_t i = 0; i < n; ++i) total += a[i];
    return total;
}

} // namespace

int main(int argc, char** argv) {
    bool quick = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quick") == 0) quick = true;
    }

    std::printf("\nvectis benchmark: vector engine vs scalar\n");
    std::printf("=========================================\n");
    {
        const auto b = cpu::brand();
        std::printf("  cpu        : %.*s\n", static_cast<int>(b.size()), b.data());
    }
    std::printf("  tier       : %s, %zu float lanes per register\n",
                to_string(compiled_isa), native_width_v<float>);

    const std::vector<std::size_t> sizes =
        quick ? std::vector<std::size_t>{4096, 262144}
              : std::vector<std::size_t>{1024, 4096, 65536, 262144, 1048576, 8388608};

    for (const std::size_t n : sizes) {
        std::vector<float> a(n), b(n), out(n);
        fill(a, 12345u);
        fill(b, 67890u);

        const double kib = static_cast<double>(n) * 4.0 / 1024.0;
        char title[160];

        vbench::options opt;
        opt.elems = n;
        opt.bytes = n * 4 * 3;
        opt.reps  = quick ? 5 : 21;

        std::snprintf(title, sizeof title,
                      "muladd: n=%zu (%.1f KiB/array, L2=%s L3=%s)",
                      n, kib,
                      (n * 4 >= 256 * 1024) ? "exceeded" : "fits",
                      (n * 4 * 3 >= 6 * 1024 * 1024) ? "exceeded" : "fits");

        auto make = [&](const char* which) {
            return [&, which]() -> double {
                if (std::strcmp(which, "scalar") == 0) {
                    return static_cast<double>(muladd_scalar(a.data(), b.data(), out.data(), n));
                }
                if (std::strcmp(which, "f32<8>") == 0) {
                    return static_cast<double>(muladd_vec<f32<8>>(a.data(), b.data(), out.data(), n));
                }
                if (std::strcmp(which, "f32<16>") == 0) {
                    return static_cast<double>(muladd_vec<f32<16>>(a.data(), b.data(), out.data(), n));
                }
                if (std::strcmp(which, "f32<32>") == 0) {
                    return static_cast<double>(muladd_vec<f32<32>>(a.data(), b.data(), out.data(), n));
                }
                return static_cast<double>(muladd_vec<native_f32>(a.data(), b.data(), out.data(), n));
            };
        };
        vbench::compare(title, opt,
                        {"scalar", "native_f32", "f32<8>", "f32<16>", "f32<32>"}, make);

        // Reduction: the case where the horizontal step is a real cost, and
        // where a wider accumulator vector stops helping as soon as the fold
        // itself is no longer the bottleneck.
        opt.bytes = n * 4;
        std::snprintf(title, sizeof title, "sum: n=%zu (%.1f KiB/array)", n, kib);
        auto make_sum = [&](const char* which) {
            return [&, which]() -> double {
                if (std::strcmp(which, "scalar") == 0) {
                    return static_cast<double>(sum_scalar(a.data(), n));
                }
                if (std::strcmp(which, "f32<8>") == 0) {
                    return static_cast<double>(sum_vec<f32<8>>(a.data(), n));
                }
                if (std::strcmp(which, "f32<32>") == 0) {
                    return static_cast<double>(sum_vec<f32<32>>(a.data(), n));
                }
                return static_cast<double>(sum_vec<native_f32>(a.data(), n));
            };
        };
        vbench::compare(title, opt, {"scalar", "native_f32", "f32<8>", "f32<32>"}, make_sum);
    }

    vbench::note("min of N runs; the scalar column is autovectorised by the compiler too");
    vbench::total_sink();
    return 0;
}
