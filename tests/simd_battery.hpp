// ===========================================================================
// Vectis - the cross-tier test battery, shared between translation units
// ===========================================================================
//
// The battery is a template over the ABI, so the same code tests scalar, AVX2
// and AVX-512.  It lives in a header because the AVX-512 instantiation cannot
// be compiled in the same translation unit as the others.
//
// WHY THAT IS: GCC declares `_mm512_cmplt_ps_mask` with a *different, reduced*
// signature when AVX512F is not enabled, so merely mentioning it - even in a
// function nobody calls - is a hard error.  The AVX-512 backend therefore
// cannot sit unexecuted inside an AVX2 binary; it needs its own translation
// unit compiled with `-mavx512f -mavx512bw -mavx512dq -mavx512vl`, entered
// only after `cpu::has(isa_level::avx512)` says the host can run it.
//
// That is also why `VECTIS_MAX_ISA_LEVEL` is not set by the build system: the
// `-m` flags already *are* the cap, and a second source of truth for the same
// decision is a bug waiting to happen.
//
// ===========================================================================
#pragma once

#include "vectis_test.hpp"

#include <vectis/core/cpu.hpp>
#include <vectis/simd/math.hpp>
#include <vectis/simd/reduce.hpp>
#include <vectis/simd/vec.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <type_traits>

namespace vtest_battery {

using namespace vectis;

// --------------------------------------------------------------- test data

/// Deterministic, awkward values: negatives, non-integers, no accidental
/// symmetry.  Deliberately not random - a failure has to be reproducible.
template <class T>
T mk(std::size_t i, int salt) {
    if constexpr (std::is_floating_point_v<T>) {
        const double v = static_cast<double>(i) * 0.37 +
                         static_cast<double>(salt) * 1.13 - 2.5;
        return static_cast<T>(v);
    } else {
        return static_cast<T>(static_cast<long long>(i) * 3 +
                              static_cast<long long>(salt) * 7 - 11);
    }
}

template <class T, std::size_t N>
std::array<T, N> make_input(int salt) {
    std::array<T, N> out{};
    for (std::size_t i = 0; i < N; ++i) out[i] = mk<T>(i, salt);
    return out;
}

/// Divisors must be non-zero: integer division by zero is UB, and float
/// division by zero takes a different path on each tier.
template <class T, std::size_t N>
std::array<T, N> make_divisor(int salt) {
    auto out = make_input<T, N>(salt);
    for (auto& v : out) {
        if (v == T{0}) v = T{1};
    }
    return out;
}

/// Whether a reduction may be compared exactly.
///
/// Integer addition is associative so the two orders agree exactly.  Floating
/// point addition is not, and no amount of care makes a pairwise fold produce
/// the same bits as a serial chain - so floats get a relative tolerance here
/// and only here.  min/max stay exact: they are associative and exact.
template <class T>
void check_reduction(T got, T want) {
    if constexpr (std::is_floating_point_v<T>) {
        const double scale = 1.0 + std::fabs(static_cast<double>(want));
        CHECK_NEAR(got, want, 1e-5 * scale);
    } else {
        CHECK_EQ(got, want);
    }
}

template <class T>
bool lanes_equal(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) return vtest::ulp_distance(a, b) == 0;
    else return a == b;
}

/// Exact for integers, bit-identical for floats (ULP distance 0).
#define CHECK_SAME(a, b)                                                       \
    do {                                                                       \
        if constexpr (std::is_floating_point_v<                                \
                          std::remove_cvref_t<decltype(a)>>) {                 \
            CHECK_ULP((a), (b), 0);                                            \
        } else {                                                               \
            CHECK_EQ((a), (b));                                                \
        }                                                                      \
    } while (0)

template <class Vn, class Vs>
void expect_same(const Vn& vn, const Vs& vs, const char* what,
                 const char* file, int line) {
    static_assert(Vn::lanes == Vs::lanes, "shape mismatch in comparison");
    const auto a = vn.to_array();
    const auto b = vs.to_array();
    for (std::size_t i = 0; i < Vn::lanes; ++i) {
        if (!lanes_equal(a[i], b[i])) {
            char sa[64], sb[64], full[256];
            vtest::format_value(sa, sizeof sa, a[i]);
            vtest::format_value(sb, sizeof sb, b[i]);
            std::snprintf(full, sizeof full,
                          "%s: lane %zu differs (%s vs oracle %s)", what, i, sa, sb);
            vtest::report_failure(full, file, line);
            return;
        }
    }
}

#define SAME(vec_expr, oracle_expr, what)                                      \
    expect_same((vec_expr), (oracle_expr), (what), __FILE__, __LINE__)

// -------------------------------------------------------------- the battery

template <class T, std::size_t N, class Abi>
void battery() {
    using Vn = basic_vec<T, N, Abi>;
    using Vs = basic_vec<T, N, scalar_abi>;

    const auto a = make_input<T, N>(1);
    const auto b = make_divisor<T, N>(2);

    const Vn va = Vn::load(a.data());
    const Vn vb = Vn::load(b.data());
    const Vs oa = Vs::load(a.data());
    const Vs ob = Vs::load(b.data());

    // ---- round trip -------------------------------------------------------
    {
        std::array<T, N> out{};
        va.store(out.data());
        for (std::size_t i = 0; i < N; ++i) CHECK_SAME(out[i], a[i]);
    }

    // ---- exact arithmetic -------------------------------------------------
    // One operation at a time, so nothing can be contracted into an FMA and the
    // two tiers must agree bit for bit.
    SAME(va + vb, oa + ob, "add");
    SAME(va - vb, oa - ob, "sub");
    if constexpr (BackendHasMul<T, Abi>) {
        SAME(va * vb, oa * ob, "mul");
    }
    if constexpr (BackendHasDiv<T, Abi>) {
        SAME(va / vb, oa / ob, "div");
    }
    if constexpr (BackendHasFma<T, Abi>) {
        SAME(va.fma(vb, va), oa.fma(ob, oa), "fma");
    }
    SAME(-va, -oa, "neg");
    SAME(va.abs(), oa.abs(), "abs");

    if constexpr (BackendHasSqrt<T, Abi>) {
        SAME(va.abs().sqrt(), oa.abs().sqrt(), "sqrt");
    }
    if constexpr (BackendHasMinMax<T, Abi>) {
        SAME(va.min(vb), oa.min(ob), "min");
        SAME(va.max(vb), oa.max(ob), "max");
    }

    // ---- bitwise ---------------------------------------------------------
    SAME(va & vb, oa & ob, "band");
    SAME(va | vb, oa | ob, "bor");
    SAME(va ^ vb, oa ^ ob, "bxor");
    SAME(andnot(va, vb), andnot(oa, ob), "andnot");

    // ---- shifts ----------------------------------------------------------
    if constexpr (BackendHasShifts<T, Abi>) {
        SAME(va << 3, oa << 3, "shl");
        SAME(va >> 3, oa >> 3, "shr_logical");
        if constexpr (std::signed_integral<T>) {
            SAME(va.shr_arith(3), oa.shr_arith(3), "shr_arith");
        }
    }

    // ---- comparisons -----------------------------------------------------
    // Bit-level comparison, which also proves that the padding lanes of a
    // partial final register never leak into bits().
    CHECK_EQ((va == vb).bits(), (oa == ob).bits());
    CHECK_EQ((va != vb).bits(), (oa != ob).bits());
    CHECK_EQ((va == vb).count(), (oa == ob).count());
    if constexpr (BackendHasOrdering<T, Abi>) {
        CHECK_EQ((va <  vb).bits(), (oa <  ob).bits());
        CHECK_EQ((va <= vb).bits(), (oa <= ob).bits());
        CHECK_EQ((va >  vb).bits(), (oa >  ob).bits());
        CHECK_EQ((va >= vb).bits(), (oa >= ob).bits());
        CHECK_EQ((va <  vb).all(),  (oa <  ob).all());
        CHECK_EQ((va <  vb).any(),  (oa <  ob).any());
        CHECK_EQ((va <  vb).none(), (oa <  ob).none());
    }

    // ---- select ----------------------------------------------------------
    {
        const auto m  = va == vb;
        const auto om = oa == ob;
        SAME(Vn::select(m, va, vb), Vs::select(om, oa, ob), "select");
    }

    // ---- mask algebra ----------------------------------------------------
    {
        using MaskT = typename Vn::mask_type;
        const auto m1 = va == vb;
        const auto m2 = va != vb;
        const auto o1 = oa == ob;
        const auto o2 = oa != ob;
        CHECK_EQ((m1 & m2).bits(), (o1 & o2).bits());
        CHECK_EQ((m1 | m2).bits(), (o1 | o2).bits());
        CHECK_EQ((m1 ^ m2).bits(), (o1 ^ o2).bits());
        CHECK_EQ((~m1).bits(),    (~o1).bits());
        CHECK_EQ(m1.count(), o1.count());
        CHECK_EQ(m1.find_first(), o1.find_first());
        for (std::size_t i = 0; i < N; ++i) CHECK_EQ(m1.test(i), o1.test(i));
        CHECK_EQ(MaskT::full().bits(), o1.lane_mask());
    }

    // ---- iota / set1 / zero ----------------------------------------------
    SAME(Vn::iota(), Vs::iota(), "iota");
    // The two-argument ramp is built from a multiply, which AVX2 lacks for
    // 64-bit lanes.  Constrained rather than emulated - see basic_vec::iota.
    if constexpr (BackendHasMul<T, Abi>) {
        SAME(Vn::iota(T{3}, T{2}), Vs::iota(T{3}, T{2}), "iota(first,step)");
    }
    SAME(Vn::set1(T{5}), Vs::set1(T{5}), "set1");
    SAME(Vn::zero(), Vs::zero(), "zero");

    // ---- reductions ------------------------------------------------------
    check_reduction(reduce_add(va), reduce_add(oa));
    if constexpr (BackendHasMinMax<T, Abi>) {
        // min/max are exact and associative, so these stay bit-for-bit.
        CHECK_SAME(reduce_min(va), reduce_min(oa));
        CHECK_SAME(reduce_max(va), reduce_max(oa));
    }
}

template <class T, class Abi>
void sweep() {
    constexpr std::size_t R = backend<T, Abi>::lanes;
    battery<T, R, Abi>();
    battery<T, R * 2, Abi>();
    battery<T, R * 3 + 2, Abi>();
    if constexpr (R > 1) {
        battery<T, R - 1, Abi>();    // partial final register
        battery<T, R + 1, Abi>();    // exactly one lane into a second register
    }
    battery<T, 3, Abi>();
}

template <class Abi>
void full_battery() {
    sweep<float, Abi>();
    sweep<double, Abi>();
    sweep<std::int32_t, Abi>();
    sweep<std::uint32_t, Abi>();
    sweep<std::int64_t, Abi>();
    sweep<std::uint64_t, Abi>();
}

// ------------------------------------------------- out-of-bounds canaries

/// A load must read exactly N lanes and a store must write exactly N.
///
/// This is the easiest bug to introduce in a partial-register vector and the
/// one that surfaces as a heap-buffer-overflow in somebody else's code, so it
/// is checked with sentinels here as well as by the sanitizer builds in CI.
template <class T, std::size_t N, class Abi>
void canary_check() {
    using V = basic_vec<T, N, Abi>;
    constexpr std::size_t pad = 16;
    constexpr T sentinel = T{1234};

    std::array<T, N + 2 * pad> buf{};
    for (auto& v : buf) v = sentinel;
    for (std::size_t i = 0; i < N; ++i) buf[pad + i] = mk<T>(i, 3);

    const V v = V::load(buf.data() + pad);

    for (std::size_t i = 0; i < pad; ++i) {
        CHECK_EQ(buf[i], sentinel);
        CHECK_EQ(buf[pad + N + i], sentinel);
    }

    std::array<T, N + 2 * pad> out{};
    for (auto& x : out) x = sentinel;
    v.store(out.data() + pad);

    for (std::size_t i = 0; i < pad; ++i) {
        CHECK_EQ(out[i], sentinel);
        CHECK_EQ(out[pad + N + i], sentinel);
    }
    for (std::size_t i = 0; i < N; ++i) CHECK_SAME(out[pad + i], buf[pad + i]);
}

template <class Abi>
void canaries() {
    canary_check<float, 1, Abi>();
    canary_check<float, 3, Abi>();
    canary_check<float, 5, Abi>();
    canary_check<float, 6, Abi>();
    canary_check<float, 7, Abi>();
    canary_check<float, 9, Abi>();
    canary_check<float, 17, Abi>();
    canary_check<double, 3, Abi>();
    canary_check<double, 5, Abi>();
    canary_check<std::int32_t, 5, Abi>();
    canary_check<std::int64_t, 3, Abi>();
    canary_check<std::int64_t, 5, Abi>();
}

// --------------------------------------------------- special value semantics

/// min(a,b) is (a<b)?a:b, NOT std::fmin.  The difference is observable with a
/// NaN in the first operand and with the sign of zero.
template <class Abi>
void nan_and_zero_semantics() {
    using V = basic_vec<float, 8, Abi>;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::array<float, 8> a{nan, 1.0f, nan, -0.0f, 0.0f, 2.0f, -1.0f, 3.0f};
    const std::array<float, 8> b{5.0f, nan, nan, 0.0f, -0.0f, -2.0f, 1.0f, 3.0f};

    const auto mn = V::load(a.data()).min(V::load(b.data())).to_array();
    const auto mx = V::load(a.data()).max(V::load(b.data())).to_array();

    for (std::size_t i = 0; i < 8; ++i) {
        const float want_min = (a[i] < b[i]) ? a[i] : b[i];
        const float want_max = (a[i] > b[i]) ? a[i] : b[i];
        CHECK_ULP(mn[i], want_min, 0);
        CHECK_ULP(mx[i], want_max, 0);
    }
}

// ------------------------------------------------------------- math accuracy

template <class Abi>
void rsqrt_accuracy(const char* tier, std::uint64_t max_ulp) {
    using V = basic_vec<float, 8, Abi>;
    std::array<float, 8> in{};
    for (std::size_t i = 0; i < 8; ++i) in[i] = 0.25f + static_cast<float>(i) * 0.75f;

    const auto got = math::rsqrt(V::load(in.data())).to_array();
    for (std::size_t i = 0; i < 8; ++i) {
        const float want = 1.0f / std::sqrt(in[i]);
        const auto d = vtest::ulp_distance(got[i], want);
        if (d > max_ulp) {
            char msg[224];
            std::snprintf(msg, sizeof msg,
                          "refined rsqrt on %s tier, lane %zu: %.9g vs %.9g "
                          "(%llu ulp, budget %llu)",
                          tier, i, static_cast<double>(got[i]),
                          static_cast<double>(want),
                          static_cast<unsigned long long>(d),
                          static_cast<unsigned long long>(max_ulp));
            vtest::report_failure(msg, __FILE__, __LINE__);
        }
    }
}

template <class Abi>
void rcp_accuracy(const char* tier, std::uint64_t max_ulp) {
    using V = basic_vec<float, 8, Abi>;
    std::array<float, 8> in{};
    for (std::size_t i = 0; i < 8; ++i) in[i] = 0.5f + static_cast<float>(i) * 1.25f;

    const auto got = math::rcp(V::load(in.data())).to_array();
    for (std::size_t i = 0; i < 8; ++i) {
        const float want = 1.0f / in[i];
        const auto d = vtest::ulp_distance(got[i], want);
        if (d > max_ulp) {
            char msg[224];
            std::snprintf(msg, sizeof msg,
                          "refined rcp on %s tier, lane %zu: %.9g vs %.9g "
                          "(%llu ulp, budget %llu)",
                          tier, i, static_cast<double>(got[i]),
                          static_cast<double>(want),
                          static_cast<unsigned long long>(d),
                          static_cast<unsigned long long>(max_ulp));
            vtest::report_failure(msg, __FILE__, __LINE__);
        }
    }
}

// ---------------------------------------------------------- padding honesty

template <class Abi>
void padding_never_observed() {
    using Vf5 = basic_vec<float, 5, Abi>;
    using Vi5 = basic_vec<std::int32_t, 5, Abi>;
    using Vf1 = basic_vec<float, 1, Abi>;

    // set1 fills every physical lane including padding - exactly the case where
    // a reduction that forgot about padding gives the wrong answer.
    CHECK_EQ(reduce_add(Vf5::set1(1.0f)), 5.0f);
    CHECK_EQ(reduce_add(Vi5::set1(1)), 5);
    CHECK_EQ(reduce_add(Vf1::set1(3.0f)), 3.0f);

    const auto m = Vf5::set1(1.0f) == Vf5::set1(1.0f);
    CHECK_EQ(m.bits(), 0x1Fu);
    CHECK_EQ(m.count(), 5u);
    CHECK(m.all());
    CHECK_EQ(Vf5::mask_type::lane_mask(), 0x1Fu);
}

// ------------------------------------------------------------ entry points

inline void run_scalar_all() {
    full_battery<scalar_abi>();
    canaries<scalar_abi>();
    nan_and_zero_semantics<scalar_abi>();
    rsqrt_accuracy<scalar_abi>("scalar", 1);
    rcp_accuracy<scalar_abi>("scalar", 1);
    padding_never_observed<scalar_abi>();
}

/// The baseline backend: whatever this translation unit was built for.
inline void run_native_all() {
    full_battery<native_abi>();
    canaries<native_abi>();
    nan_and_zero_semantics<native_abi>();
    padding_never_observed<native_abi>();
}

} // namespace vtest_battery
