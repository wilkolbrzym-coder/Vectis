// ===========================================================================
// Vectis - core tests (portable translation unit)
// ===========================================================================
//
// Covers the scalar oracle, the native tier, and AVX2.  The AVX-512 battery
// lives in test_simd_avx512.cpp, compiled with AVX-512 enabled - see the
// explanation at the top of simd_battery.hpp for why it cannot live here.
//
// ===========================================================================
#include "backend_batteries.hpp"
#include "simd_battery.hpp"
#include "vectis_test.hpp"

#include <vectis/simd/math.hpp>
#include <vectis/simd/reduce.hpp>
#include <vectis/simd/vec.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace vectis;
using vtest_battery::mk;

// ===========================================================================
// Tiers
// ===========================================================================

VECTIS_TEST(simd_core_scalar_oracle) {
    vtest_battery::run_scalar_all();
}

VECTIS_TEST(simd_core_native) {
    vtest_battery::run_native_all();
}

// ===========================================================================
// Shape arithmetic
// ===========================================================================

VECTIS_TEST(simd_shape_arithmetic) {
    // These are the claims the design rests on, so they are static_asserts
    // rather than runtime checks: they hold or the code does not compile.
    // The AVX-512 shape asserts live in test_simd_avx512.cpp: this translation
    // unit is not compiled with AVX-512 enabled, so that backend does not exist
    // here at all.
    static_assert(basic_vec<float, 3, scalar_abi>::num_regs == 3);
    static_assert(basic_vec<float, 3, scalar_abi>::tail_lanes == 0);

    CHECK_EQ(f32<16>::lanes, 16u);
    CHECK_EQ(f32<16>::num_regs, 16u / native_width_v<float>);
    CHECK_EQ(native_f32::lanes, native_width_v<float>);
    CHECK_EQ(native_f64::lanes, native_width_v<double>);
    CHECK_EQ(native_i32::lanes, native_width_v<std::int32_t>);
}

// ===========================================================================
// The concepts are load-bearing, so they get a test too
// ===========================================================================

VECTIS_TEST(simd_concepts_hold) {
    static_assert(SimdVec<f32<8>>);
    static_assert(SimdVec<native_f32>);
    static_assert(SimdVec<basic_vec<double, 4, scalar_abi>>);
    static_assert(SimdMask<f32<8>::mask_type, f32<8>>);
    static_assert(SameShape<f32<8>, basic_vec<float, 8, scalar_abi>>);
    static_assert(VecOf<f32<8>, float>);
    static_assert(!VecOf<f32<8>, double>);
    static_assert(Vectorizable<float>);
    static_assert(!Vectorizable<bool>);
    static_assert(FloatLane<double>);

    // Capability probes: how a kernel says "only where the hardware can do
    // this" instead of hard-coding an ISA check.  The AVX2 and AVX-512
    // capability assertions live in their own translation units, because
    // neither backend exists in this one.
    // The scalar oracle is complete for every lane type, by construction.
    static_assert(BackendHasMul<std::int64_t, scalar_abi>);
    static_assert(BackendHasMinMax<std::uint64_t, scalar_abi>);
    CHECK(true);
}

// ===========================================================================
// Vector algebra
// ===========================================================================

VECTIS_TEST(simd_vector_algebra) {
    constexpr std::size_t N = 16;
    std::array<float, N> a{}, b{};
    for (std::size_t i = 0; i < N; ++i) {
        a[i] = 1.0f + static_cast<float>(i) * 0.25f;
        b[i] = 2.0f - static_cast<float>(i) * 0.125f;
    }

    using V = f32<N>;
    const V va = V::load(a.data());
    const V vb = V::load(b.data());

    double want_dot = 0.0, want_len2 = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        want_dot  += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        want_len2 += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    }
    CHECK_NEAR(dot(va, vb), want_dot, 1e-3);
    CHECK_NEAR(length_sq(va), want_len2, 1e-3);
    CHECK_NEAR(length(va), std::sqrt(want_len2), 1e-4);

    const auto n = normalize(va).to_array();
    for (std::size_t i = 0; i < N; ++i) {
        const float want_n =
            static_cast<float>(a[i]) / std::sqrt(static_cast<float>(want_len2));
        CHECK_NEAR(n[i], want_n, 1e-5);
    }
}

// ===========================================================================
// Math shaping
// ===========================================================================

VECTIS_TEST(simd_math_clamp_lerp_saturate) {
    constexpr std::size_t N = 16;
    std::array<float, N> a{}, b{}, t{};
    for (std::size_t i = 0; i < N; ++i) {
        a[i] = mk<float>(i, 1);
        b[i] = mk<float>(i, 4);
        t[i] = mk<float>(i, 9);
    }

    using V = f32<N>;
    const V va = V::load(a.data());
    const V vb = V::load(b.data());
    const V vt = V::load(t.data());

    const auto sat = math::saturate(vt).to_array();
    const auto cl  = math::clamp(vt, -0.5f, 0.5f).to_array();
    const auto lp  = math::lerp(va, vb, math::saturate(vt)).to_array();

    for (std::size_t i = 0; i < N; ++i) {
        CHECK_ULP(sat[i], std::fmin(std::fmax(t[i], 0.0f), 1.0f), 0);
        CHECK_ULP(cl[i], std::fmin(std::fmax(t[i], -0.5f), 0.5f), 0);
        const float ts = std::fmin(std::fmax(t[i], 0.0f), 1.0f);
        CHECK_NEAR(lp[i], a[i] + ts * (b[i] - a[i]), 1e-5);
    }

    // lerp is exact at both ends - the whole reason for the fma form.
    CHECK_ULP(math::lerp(va, vb, V::zero()).to_array()[0], a[0], 0);
    CHECK_ULP(math::lerp(va, vb, V::set1(1.0f)).to_array()[0], b[0], 0);
}

VECTIS_TEST(simd_math_sign_and_abs_diff) {
    std::array<float, 8> in{-3.0f, -0.0f, 0.0f, 0.5f, 7.0f, -1e-30f, 1e30f, -2.0f};
    using V = f32<8>;
    const auto s = math::sign(V::load(in.data())).to_array();
    const auto d = math::abs_diff(V::load(in.data()), V::zero()).to_array();
    for (std::size_t i = 0; i < 8; ++i) {
        const float want = (in[i] > 0.0f) ? 1.0f : ((in[i] < 0.0f) ? -1.0f : 0.0f);
        CHECK_ULP(s[i], want, 0);
        CHECK_ULP(d[i], std::fabs(in[i]), 0);
    }
}

// ===========================================================================
// Chunked iteration: the shape a real loop has, ragged tail and all
// ===========================================================================

VECTIS_TEST(simd_for_each_chunk_handles_ragged_tail) {
    for (std::size_t n = 0; n <= 40; ++n) {
        std::vector<float> in(n), out(n, 0.0f);
        for (std::size_t i = 0; i < n; ++i) in[i] = static_cast<float>(i) + 0.5f;

        // The indexed form is the one a kernel needs: it has to know where in
        // the output array it is writing.
        for_each_chunk_indexed<float, 8>(
            in.data(), n,
            [&](f32<8> v, std::size_t off) { (v * v).store(out.data() + off); },
            [&](float x, std::size_t off) { out[off] = x * x; });

        for (std::size_t i = 0; i < n; ++i) CHECK_NEAR(out[i], in[i] * in[i], 0.0);
    }

    // The plain form must visit every element exactly once, in order.
    {
        std::vector<float> in(37);
        for (std::size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(i);
        double vec_sum = 0.0, tail_sum = 0.0;
        std::size_t vec_count = 0, tail_count = 0;
        for_each_chunk<float, 8>(
            in.data(), in.size(),
            [&](f32<8> v) {
                vec_sum += static_cast<double>(reduce_add(v));
                vec_count += 8;
            },
            [&](float x) { tail_sum += static_cast<double>(x); ++tail_count; });
        CHECK_EQ(vec_count + tail_count, in.size());
        CHECK_EQ(vec_count, 32u);
        CHECK_EQ(tail_count, 5u);
        CHECK_NEAR(vec_sum + tail_sum, 36.0 * 37.0 / 2.0, 1e-3);
    }
}

// ===========================================================================
// The ISA-specific batteries, entered only from here
// ===========================================================================
//
// The capability check lives in THIS translation unit, which is compiled for
// the portable tier.  It has to: a TU compiled with -mavx512f may contain
// AVX-512 instructions in every function it defines, including a function that
// only asks whether AVX-512 is available - so a guard inside such a TU can
// fault before it returns false.  Asking here and crossing into that TU through
// a call is the only ordering that is actually safe.

namespace {

void report_skip(const char* tier, bool built, bool supported) {
    char msg[128];
    if (!built) {
        std::snprintf(msg, sizeof msg,
                      "the %s battery was not compiled into this build", tier);
        vtest::skip(msg);
    } else if (!supported) {
        std::snprintf(msg, sizeof msg,
                      "host has no %s (compiled and linked, never executed here)",
                      tier);
        vtest::skip(msg);
    }
}

} // namespace

VECTIS_TEST(simd_core_avx2) {
    const bool built = vtest_batteries::avx2_battery_built();
    const bool ok    = cpu::has(isa_level::avx2);
    report_skip("AVX2", built, ok);
    if (built && ok) vtest_batteries::run_avx2();
}

VECTIS_TEST(simd_core_avx512) {
    const bool built = vtest_batteries::avx512_battery_built();
    const bool ok    = cpu::has(isa_level::avx512);
    report_skip("AVX-512", built, ok);
    if (built && ok) vtest_batteries::run_avx512();
}

// ===========================================================================
// Canaries, as their own test names so `ctest -R canary` isolates them
// ===========================================================================

VECTIS_TEST(simd_canary_scalar) {
    vtest_battery::canaries<scalar_abi>();
}

VECTIS_TEST(simd_canary_native) {
    vtest_battery::canaries<native_abi>();
}

VECTIS_TEST(simd_canary_avx512) {
    if (!vtest_batteries::avx512_battery_built() || !cpu::has(isa_level::avx512)) {
        vtest::skip("no AVX-512 here");
        return;
    }
    vtest_batteries::run_avx512_canaries();
}
