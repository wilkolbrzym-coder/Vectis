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
#include <vector>

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
        // Bit-for-bit is the right expectation here only because this data
        // holds no NaN and no signed-zero tie, and those are the sole cases in
        // which a min/max reduction is order-independent.  The cases that are
        // not are the point of reduce_agrees_with_oracle, which runs the same
        // shapes with both present.
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
///
/// CHECK_BITS, not CHECK_ULP: `ulp_distance` has to report `+0.0` and `-0.0` as
/// zero apart to be a useful distance for ordinary values, which made the two
/// `±0` lanes below - the ones this rule exists to distinguish - pass whichever
/// zero came back.  Only a bit-pattern comparison tests it.
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
        CHECK_BITS(mn[i], want_min);
        CHECK_BITS(mx[i], want_max);
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

// ===========================================================================
// Regression batteries
//
// Every check below exists because the bug it catches shipped, or was one
// commit away from shipping, with a green suite.  They are grouped here rather
// than folded into the battery above so the reason each one exists stays
// readable, and so a failure names the property that broke.
// ===========================================================================

/// Bit patterns worth feeding a mask factory: exhaustive when the shape is
/// small enough to be, a spread of awkward ones when it is not.
template <std::size_t N>
std::vector<std::uint64_t> mask_patterns() {
    const std::uint64_t all =
        (N >= 64) ? ~std::uint64_t{0} : ((std::uint64_t{1} << N) - 1);
    std::vector<std::uint64_t> out;
    if constexpr (N <= 8) {
        for (std::uint64_t b = 0; b <= all; ++b) out.push_back(b);
    } else {
        out.push_back(0);
        out.push_back(all);
        out.push_back(all / 3);
        out.push_back(0xAAAA'AAAA'AAAA'AAAAull & all);
        out.push_back(0x5555'5555'5555'5555ull & all);
        for (std::size_t i = 0; i < N; ++i) out.push_back(std::uint64_t{1} << i);
    }
    return out;
}

/// Every mask factory must return exactly the predicate it was asked for, at
/// every lane, on every tier.
///
/// This battery exists because it did not: the AVX2 backend shifted the
/// broadcast bit pattern the wrong way, so `from_bits` returned a mask with at
/// most lane 0 set and `from_lane(1)` returned nothing at all.  254 of 256
/// eight-bit patterns were wrong and the suite was green, because the only
/// assertions on these factories lived in the AVX-512 translation unit - which
/// is skipped on any host that has no AVX-512.  The factories are
/// ABI-independent by contract, so the check belongs here, where every tier
/// inherits it and a host without AVX-512 still runs it.
template <class T, std::size_t N, class Abi>
void mask_factory_check() {
    using M = basic_mask<T, N, Abi>;
    static_assert(N <= 64, "mask_factory_check assumes bits() fits one word");

    const std::uint64_t all =
        (N >= 64) ? ~std::uint64_t{0} : ((std::uint64_t{1} << N) - 1);

    for (const std::uint64_t b : mask_patterns<N>()) {
        CHECK_EQ(M::from_bits(b).bits(), b);
    }

    // Each single lane, both directions: what the factory says and what the
    // observer then reports.
    for (std::size_t i = 0; i < N; ++i) {
        const M one = M::from_lane(i);
        CHECK_EQ(one.bits(), std::uint64_t{1} << i);
        CHECK(one.test(i));
        CHECK_EQ(one.count(), 1u);
        CHECK(one.any());
        CHECK(!one.none());
    }

    CHECK_EQ(M::full().bits(), all);
    CHECK(M::full().all());
    CHECK_EQ(M::full().count(), N);
    CHECK(M::from_bits(all).all());
    CHECK(M::from_lane(N).none());        // out of range: an empty mask
    CHECK_EQ(M::from_bits(0).count(), 0u);
}

template <class Abi>
void mask_factories() {
    mask_factory_check<float, 8, Abi>();
    mask_factory_check<float, 6, Abi>();        // partial final register
    mask_factory_check<float, 3, Abi>();
    mask_factory_check<float, 1, Abi>();
    mask_factory_check<double, 4, Abi>();
    mask_factory_check<std::int32_t, 5, Abi>();
    mask_factory_check<std::uint64_t, 3, Abi>();
}

/// `operator==` on a mask must mean what every other observer means.
///
/// The padding lanes of a partially-filled final register hold a real
/// predicate, so comparing the raw registers lets two masks that agree on every
/// lane the caller can see - same bits(), same count(), same test(i) for every
/// i - compare unequal.  A mask built by a comparison seeds its padding with
/// the comparison's result; one built by from_bits leaves it clear.  Both are
/// the "all lanes true" mask of a six-lane vector.
template <class Abi>
void mask_equality_is_observable_equality() {
    using V = basic_vec<float, 6, Abi>;         // 6 of 8 lanes when R == 8
    using M = typename V::mask_type;

    const std::array<float, 6> in{0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
    const V v = V::load(in.data());

    const M from_cmp = (v == v);
    const M from_fac = M::from_bits(M::lane_mask());

    CHECK_EQ(from_cmp.bits(), from_fac.bits());
    CHECK_EQ(from_cmp.count(), from_fac.count());
    for (std::size_t i = 0; i < 6; ++i) {
        CHECK_EQ(from_cmp.test(i), from_fac.test(i));
    }
    CHECK(from_cmp == from_fac);
    CHECK(!(from_cmp != from_fac));

    const M low3 = (v < V::set1(3.0f));
    CHECK_EQ(low3.bits(), 0x07u);
    CHECK(low3 == M::from_bits(0x07u));
    CHECK(low3 != M::from_bits(0x08u));
    CHECK(!(low3 == M::from_bits(0x08u)));
}

/// reduce_min/reduce_max must return the same answer at every tier for the same
/// logical input.  The scalar backend is the oracle, so anything else is a bug
/// in the fold order.
///
/// The rule behind both is `(a<b) ? a : b`, which is neither associative nor
/// commutative: with a NaN or a signed zero present the answer depends on where
/// the operands sit.  Two wrong versions hid behind that.  The first kept the
/// *left* operand on a tie (`candidate < current`), so
/// `reduce_min([-0.0, +0.0, ...])` was `-0.0` on AVX2 and `+0.0` on scalar.  The
/// second was the fold order itself: the registers were collapsed lane-wise into
/// one accumulator and scanned afterwards, which reorders every element after a
/// NaN, so the 16-lane AVX2 vector `{100,50,60,0.5,70,80,90,95,NaN,10,20,30,40,
/// 45,55,65}` reduced to `0.5` on AVX2 and to `10` on scalar.  CHECK_BITS, not
/// CHECK_ULP, because the sign of zero is the whole point and ulp_distance
/// deliberately reports it as zero.
template <class Abi, std::size_t N>
void reduce_oracle_at() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();

    // Eight lanes each, cycled to fill N.  The cycling is the load-bearing part:
    // it is what puts a NaN or a signed zero in more than one register, and a
    // vector that fits in a single register takes the same path as the oracle
    // does, so it can never see the difference.
    const std::array<std::array<float, 8>, 9> seeds{{
        {nan, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f},        // NaN first
        {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, nan},        // NaN last
        {1.f, nan, 2.f, 3.f, nan, 5.f, 6.f, 7.f},        // NaN in the middle
        {nan, nan, nan, nan, nan, nan, nan, nan},
        {-0.0f, 0.0f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f},     // -0 vs +0 tie
        {0.0f, -0.0f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f},
        {-0.0f, -0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {-inf, inf, 0.f, -1.f, 2.f, -3.f, 4.f, -5.f},
        {3.f, 3.f, 3.f, 3.f, 3.f, 3.f, 3.f, 3.f},        // every lane equal
    }};

    for (const auto& seed : seeds) {
        std::array<float, N> in{};
        for (std::size_t i = 0; i < N; ++i) in[i] = seed[i % 8];

        const auto vn = basic_vec<float, N, Abi>::load(in.data());
        const auto vs = basic_vec<float, N, scalar_abi>::load(in.data());
        CHECK_BITS(reduce_min(vn), reduce_min(vs));
        CHECK_BITS(reduce_max(vn), reduce_max(vs));
    }

    // The seeds above are periodic, and a periodic vector folds to the same
    // shape it started as - so they cannot see a fold-order bug no matter how
    // many NaNs they contain.  Seeing one takes a strictly monotone ramp with a
    // NaN at a single position, which is the arrangement where the extreme value
    // sits on opposite sides of the NaN in the two orders: on a rising ramp that
    // is the minimum (lane order meets it first, column order meets it after the
    // NaN), and on a falling ramp it is the maximum.  Sweeping the NaN across
    // every position covers both.
    for (std::size_t q = 0; q < N; ++q) {
        for (int falling = 0; falling < 2; ++falling) {
            std::array<float, N> in{};
            for (std::size_t i = 0; i < N; ++i) {
                const auto step = static_cast<float>(i);
                in[i] = falling != 0 ? -step : step;
            }
            in[q] = nan;

            const auto vn = basic_vec<float, N, Abi>::load(in.data());
            const auto vs = basic_vec<float, N, scalar_abi>::load(in.data());
            CHECK_BITS(reduce_min(vn), reduce_min(vs));
            CHECK_BITS(reduce_max(vn), reduce_max(vs));
        }
    }
}

template <class Abi>
void reduce_agrees_with_oracle() {
    constexpr std::size_t R = backend<float, Abi>::lanes;
    // The shapes sweep() uses, and for the same reason: the multi-register path
    // and the partial final register are both different code, and testing only
    // one full register is what let the fold-order bug through.
    reduce_oracle_at<Abi, R>();
    reduce_oracle_at<Abi, R * 2>();
    reduce_oracle_at<Abi, R * 3 + 2>();
    if constexpr (R > 1) {
        reduce_oracle_at<Abi, R - 1>();    // partial final register
        reduce_oracle_at<Abi, R + 1>();    // exactly one lane into a second
    }
}

/// Integer add/sub/mul/neg wrap at the boundary values, as the hardware does.
///
/// The scalar backend used to compute these with ordinary signed arithmetic,
/// where overflow is UB - so the oracle the vector backends are compared
/// against had undefined behaviour exactly where it mattered most, at INT_MIN
/// and INT_MAX.  Nothing observed it until UBSan did.  These are the inputs
/// that make it visible, and they run on the scalar tier too.
///
/// Driven through the vector interface, not through the backend's static
/// functions: a register is a `__m256i` on AVX2 and a plain `T` on the scalar
/// tier, so `backend<T,Abi>::add(hi, one)` is not even expressible.  One lane
/// holds the boundary value and lane 0 is what gets read.
template <class T, class Abi>
void integer_wrap_check() {
    static_assert(std::is_integral_v<T>);
    using V = basic_vec<T, 2, Abi>;
    using U = std::make_unsigned_t<T>;

    constexpr T lo = std::numeric_limits<T>::min();
    constexpr T hi = std::numeric_limits<T>::max();

    const auto lane0 = [](const V& v) { return v.to_array()[0]; };

    CHECK_EQ(lane0(V::set1(hi) + V::set1(T{1})), lo);
    CHECK_EQ(lane0(V::set1(lo) - V::set1(T{1})), hi);
    CHECK_EQ(lane0(V::set1(lo) + V::set1(lo)), T{0});
    CHECK_EQ(lane0(-V::set1(lo)), lo);            // min is its own negation
    CHECK_EQ(lane0(V::zero() - V::set1(lo)), lo);

    if constexpr (BackendHasMul<T, Abi>) {
        CHECK_EQ(lane0(V::set1(hi) * V::set1(T{2})),
                 static_cast<T>(static_cast<U>(hi) * U{2}));
        CHECK_EQ(lane0(V::set1(lo) * V::set1(static_cast<T>(U{0} - U{1}))), lo);
    }
}

template <class Abi>
void integer_wrap_semantics() {
    integer_wrap_check<std::int32_t, Abi>();
    integer_wrap_check<std::uint32_t, Abi>();
    integer_wrap_check<std::int64_t, Abi>();
    integer_wrap_check<std::uint64_t, Abi>();
}

/// The endpoints of the refined reciprocal functions.
///
/// The Newton step is only allowed to improve the estimate.  It did not:
/// `x = 0` seeds `+inf` and `x = +inf` seeds `0`, and `y * (1.5 - 0.5*x*y*y)`
/// turns both into a NaN, so `rsqrt(0)` - and therefore `normalize` of a zero
/// vector - returned NaN while the un-refined estimate one line above was
/// exactly right.  The finite cases are covered by the accuracy batteries; this
/// is the boundary they leave out.
template <class Abi>
void reciprocal_endpoints() {
    using V = basic_vec<float, 4, Abi>;
    const float inf = std::numeric_limits<float>::infinity();

    const auto rs_zero = math::rsqrt(V::set1(0.0f)).to_array();
    const auto rs_inf  = math::rsqrt(V::set1(inf)).to_array();
    const auto rc_zero = math::rcp(V::set1(0.0f)).to_array();
    const auto rc_inf  = math::rcp(V::set1(inf)).to_array();

    for (std::size_t i = 0; i < 4; ++i) {
        CHECK_BITS(rs_zero[i], inf);
        CHECK_BITS(rs_inf[i], 0.0f);
        CHECK_BITS(rc_zero[i], inf);
        CHECK_BITS(rc_inf[i], 0.0f);
    }
}

/// A vector built lane by lane must hold exactly those lanes.
///
/// The constructor loaded a whole register out of an `std::array<T, N>` for
/// every register, so a vector whose lane count does not fill its last register
/// read past the end of that array - a stack over-read of `reg_lanes -
/// tail_lanes` elements, measured by AddressSanitizer, with the garbage landing
/// in the padding lanes where it also made the mask comparison above
/// non-deterministic.  This is the value-level half of that check; the canaries
/// cover the load/store half.
template <class Abi>
void per_lane_construction() {
    {
        const basic_vec<float, 6, Abi> v{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        const auto a = v.to_array();
        for (std::size_t i = 0; i < 6; ++i) {
            CHECK_EQ(a[i], static_cast<float>(i) + 1.0f);
        }
    }
    {
        const auto v = basic_vec<std::int32_t, 5, Abi>{10, 20, 30, 40, 50};
        const auto a = v.to_array();
        for (std::size_t i = 0; i < 5; ++i) {
            CHECK_EQ(a[i], static_cast<std::int32_t>(i + 1) * 10);
        }
    }
    {
        // A lane count that is not even a whole number of registers.
        const auto v = basic_vec<double, 3, Abi>{0.5, 1.5, 2.5};
        const auto a = v.to_array();
        CHECK_EQ(a[0], 0.5);
        CHECK_EQ(a[1], 1.5);
        CHECK_EQ(a[2], 2.5);
    }
}

// ------------------------------------------------------------ entry points

inline void run_scalar_all() {
    full_battery<scalar_abi>();
    canaries<scalar_abi>();
    nan_and_zero_semantics<scalar_abi>();
    rsqrt_accuracy<scalar_abi>("scalar", 1);
    rcp_accuracy<scalar_abi>("scalar", 1);
    padding_never_observed<scalar_abi>();
    mask_factories<scalar_abi>();
    mask_equality_is_observable_equality<scalar_abi>();
    reduce_agrees_with_oracle<scalar_abi>();
    integer_wrap_semantics<scalar_abi>();
    reciprocal_endpoints<scalar_abi>();
    per_lane_construction<scalar_abi>();
}

/// The baseline backend: whatever this translation unit was built for.
inline void run_native_all() {
    full_battery<native_abi>();
    canaries<native_abi>();
    nan_and_zero_semantics<native_abi>();
    padding_never_observed<native_abi>();
    mask_factories<native_abi>();
    mask_equality_is_observable_equality<native_abi>();
    reduce_agrees_with_oracle<native_abi>();
    integer_wrap_semantics<native_abi>();
    reciprocal_endpoints<native_abi>();
    per_lane_construction<native_abi>();
}

} // namespace vtest_battery
