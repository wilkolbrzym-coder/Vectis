// ===========================================================================
// Vectis - assembly / intrinsics parity
// ===========================================================================
//
// The assembly layer only earns its complexity if it computes exactly what the
// intrinsics compute.  Not "close" - identical, bit for bit, including on the
// ragged tail where the two implementations have different loop structures and
// therefore different summation orders.
//
// That last point is the reason this test exists and is not paranoia: a
// four-accumulator dot product sums in a different order than a one-accumulator
// one, and floating-point addition is not associative.  So the *scalar
// reference* is compared with a tolerance, while asm and intrinsics - which
// share the accumulator shape by construction - are compared exactly.
//
// ===========================================================================
#include "vectis_test.hpp"

#include <vectis/core/cpu.hpp>
#include <vectis/kernels/asm_kernels.hpp>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace vectis;

#if defined(VECTIS_HAS_ASM_KERNELS) && defined(VECTIS_ARCH_X86)

namespace {

void fill(std::vector<float>& v, std::uint32_t seed) {
    std::uint32_t s = seed;
    for (auto& x : v) {
        s = s * 1664525u + 1013904223u;
        x = static_cast<float>(static_cast<double>(s >> 8) / 8388608.0 - 1.0);
    }
}

/// Every length from 0 to 200 exercises every tail residue against the 32-float
/// unrolled body, plus a few sizes that cross the 32-float boundary repeatedly.
std::vector<std::size_t> interesting_sizes() {
    std::vector<std::size_t> sizes;
    for (std::size_t n = 0; n <= 200; ++n) sizes.push_back(n);
    for (std::size_t n : {255u, 256u, 257u, 511u, 512u, 1000u, 4096u, 4097u}) {
        sizes.push_back(n);
    }
    return sizes;
}

} // namespace

VECTIS_TEST(asm_dot_matches_intrinsics_exactly) {
    if (!kernels::asm_kernels_available()) {
        vtest::skip("built without assembly kernels");
        return;
    }
    if (!kernels::runnable_here()) {
        vtest::skip("host has no AVX2");
        return;
    }

    constexpr std::size_t max_n = 4097;
    std::vector<float> a(max_n), b(max_n);
    fill(a, 1u);
    fill(b, 2u);

    for (const std::size_t n : interesting_sizes()) {
        const float from_asm = vectis_asm_dot_f32(a.data(), b.data(), n);
        const float from_int = kernels::dot_f32_intrinsics(a.data(), b.data(), n);
        const float from_sca = kernels::dot_f32_scalar(a.data(), b.data(), n);

        // Exact between asm and intrinsics: same accumulator shape, same order.
        if (vtest::ulp_distance(from_asm, from_int) != 0) {
            char msg[224];
            std::snprintf(msg, sizeof msg,
                          "dot n=%zu: asm %.9g vs intrinsics %.9g (%llu ulp)",
                          n, static_cast<double>(from_asm),
                          static_cast<double>(from_int),
                          static_cast<unsigned long long>(
                              vtest::ulp_distance(from_asm, from_int)));
            vtest::report_failure(msg, __FILE__, __LINE__);
        }
        // The scalar reference sums in one chain, so it is only close.
        // Written as a double throughout: clang resolves std::fabs on a float to
        // the C fabs(double) overload and then reports the widening, which with
        // -Wdouble-promotion and -Werror fails the build.
        const double scale = 1.0 + std::fabs(static_cast<double>(from_sca));
        CHECK_NEAR(from_asm, from_sca, 1e-2 * scale);
    }
}

VECTIS_TEST(asm_saxpy_matches_intrinsics_exactly) {
    if (!kernels::asm_kernels_available()) {
        vtest::skip("built without assembly kernels");
        return;
    }
    if (!kernels::runnable_here()) {
        vtest::skip("host has no AVX2");
        return;
    }

    constexpr std::size_t max_n = 4097;
    constexpr float alpha = 1.5f;
    std::vector<float> x(max_n), y_asm(max_n), y_int(max_n), y_sca(max_n);
    fill(x, 7u);

    for (const std::size_t n : interesting_sizes()) {
        fill(y_asm, 11u);
        y_int = y_asm;
        y_sca = y_asm;

        vectis_asm_saxpy_f32(y_asm.data(), x.data(), alpha, n);
        kernels::saxpy_f32_intrinsics(y_int.data(), x.data(), alpha, n);
        kernels::saxpy_f32_scalar(y_sca.data(), x.data(), alpha, n);

        for (std::size_t i = 0; i < n; ++i) {
            if (vtest::ulp_distance(y_asm[i], y_int[i]) != 0) {
                char msg[224];
                std::snprintf(msg, sizeof msg,
                              "saxpy n=%zu lane %zu: asm %.9g vs intrinsics %.9g",
                              n, i, static_cast<double>(y_asm[i]),
                              static_cast<double>(y_int[i]));
                vtest::report_failure(msg, __FILE__, __LINE__);
                break;
            }
            // Exact, because all three implementations now fuse explicitly:
            // the asm tail ends in vfmadd231ss, the intrinsics twin uses
            // std::fma, and so does the scalar reference.  A correctly rounded
            // fused multiply-add is the same value everywhere, so there is no
            // tolerance to choose here - and choosing one is how the previous
            // version of this test came to fail at -O0 and pass at -O2.
            CHECK_ULP(y_asm[i], y_sca[i], 0);
        }

        // Nothing past n may be touched: the tail loop is the easy place to
        // write one element too many.
        for (std::size_t i = n; i < max_n; ++i) {
            if (vtest::ulp_distance(y_asm[i], y_int[i]) != 0) {
                char msg[224];
                std::snprintf(msg, sizeof msg,
                              "saxpy n=%zu: wrote past the end at lane %zu", n, i);
                vtest::report_failure(msg, __FILE__, __LINE__);
                break;
            }
        }
    }
}

VECTIS_TEST(asm_sum_matches_intrinsics_exactly) {
    if (!kernels::asm_kernels_available()) {
        vtest::skip("built without assembly kernels");
        return;
    }
    if (!kernels::runnable_here()) {
        vtest::skip("host has no AVX2");
        return;
    }

    constexpr std::size_t max_n = 4097;
    std::vector<float> a(max_n);
    fill(a, 5u);

    for (const std::size_t n : interesting_sizes()) {
        const float from_asm = vectis_asm_sum_f32(a.data(), n);
        const float from_int = kernels::sum_f32_intrinsics(a.data(), n);
        const float from_sca = kernels::sum_f32_scalar(a.data(), n);

        const auto d = vtest::ulp_distance(from_asm, from_int);
        if (d != 0) {
            char msg[224];
            std::snprintf(msg, sizeof msg,
                          "sum n=%zu: asm %.9g vs intrinsics %.9g (%llu ulp)",
                          n, static_cast<double>(from_asm),
                          static_cast<double>(from_int),
                          static_cast<unsigned long long>(d));
            vtest::report_failure(msg, __FILE__, __LINE__);
        }
        const double scale = 1.0 + std::fabs(static_cast<double>(from_sca));
        CHECK_NEAR(from_asm, from_sca, 1e-2 * scale);
    }
}

#else

VECTIS_TEST(asm_kernels_not_built) {
    vtest::skip("this build has no assembly layer "
                "(VECTIS_ENABLE_ASM=OFF or non-x86 host)");
}

#endif
