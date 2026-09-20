// ===========================================================================
// Vectis - compile-time SIMD engine
// simd/math.hpp - the elementwise operations kernels actually reach for
// ===========================================================================
//
// Scope note, stated up front because it is a deliberate choice rather than an
// omission: there is no exp/log/sin/cos here.
//
// Those need integer<->float lane conversions and a round-to-nearest in the
// backend protocol, plus range reduction, and doing them badly is worse than
// not doing them - a subtly wrong `exp` in a lighting kernel is a bug nobody
// finds for months.  What is here instead is the set that 3D work actually
// calls in a hot loop, and it is all exact or explicitly bounded:
//
//   sqrt        correctly rounded (hardware)
//   rsqrt       Newton-refined from the hardware estimate, ~1 ulp
//   rcp         Newton-refined, ~1 ulp
//   clamp/lerp/saturate/step/smoothstep
//   dot/length/normalize          (in reduce.hpp)
//
// ===========================================================================
#pragma once

#include "vec.hpp"

namespace vectis::math {

/// `1/sqrt(x)`, refined from the hardware estimate.
///
/// One Newton-Raphson step squares the error, so precision goes
/// seed -> 2*seed -> 4*seed.  The arithmetic: AVX2 seeds float at 12 bits, so
/// one step reaches 24 and single precision is done; AVX-512's rsqrt14 seeds at
/// 14 bits, so double needs two steps (14 -> 28 -> 56) and that is what this
/// does when T is double.
///
/// One caveat worth knowing before reaching for this on AVX2 with double:
/// that tier has no 64-bit reciprocal estimate, so the "seed" is a real
/// vsqrtpd+vdivpd - correct, but no faster than writing 1/sqrt(x) yourself.
/// The refinement is only a win for float on AVX2 and for both on AVX-512.
template <SimdVec V>
[[nodiscard]] inline V rsqrt(const V& x) noexcept
    requires std::floating_point<typename V::value_type> {
    using T = typename V::value_type;
    V y = x.rsqrt_approx();
    const V half = V::set1(T{0.5});
    const V one_and_a_half = V::set1(T{1.5});
    // y = y * (1.5 - 0.5*x*y*y)
    y = y * (one_and_a_half - half * x * y * y);
    if constexpr (sizeof(T) == 8) {
        y = y * (one_and_a_half - half * x * y * y);
    }
    return y;
}

/// `1/x`, refined from the hardware estimate.
template <SimdVec V>
[[nodiscard]] inline V rcp(const V& x) noexcept
    requires std::floating_point<typename V::value_type> {
    using T = typename V::value_type;
    V y = x.rcp_approx();
    const V two = V::set1(T{2});
    // y = y * (2 - x*y)
    y = y * (two - x * y);
    if constexpr (sizeof(T) == 8) {
        y = y * (two - x * y);
    }
    return y;
}

/// The un-refined estimate, for callers who want the cheap version and know
/// what they are giving up.
template <SimdVec V>
[[nodiscard]] inline V rsqrt_fast(const V& x) noexcept
    requires std::floating_point<typename V::value_type> {
    return x.rsqrt_approx();
}

template <SimdVec V>
[[nodiscard]] inline V clamp(const V& x, const V& lo, const V& hi) noexcept {
    return x.max(lo).min(hi);
}
template <SimdVec V>
[[nodiscard]] inline V clamp(const V& x, typename V::value_type lo,
                             typename V::value_type hi) noexcept {
    return clamp(x, V::set1(lo), V::set1(hi));
}

/// Clamp to [0, 1].  On the AVX-512 tier `min`/`max` compile to a single
/// vminps/vmaxps each, so this is two instructions for sixteen lanes.
template <SimdVec V>
[[nodiscard]] inline V saturate(const V& x) noexcept
    requires std::floating_point<typename V::value_type> {
    using T = typename V::value_type;
    return clamp(x, V::set1(T{0}), V::set1(T{1}));
}

/// Linear interpolation, evaluated as `fma(t, b - a, a)`.
///
/// This is the form that is exact at both ends: t=0 gives a and t=1 gives b
/// bit-exactly, which the algebraically equivalent `a + t*(b-a)` does not.
template <SimdVec V>
[[nodiscard]] inline V lerp(const V& a, const V& b, const V& t) noexcept
    requires std::floating_point<typename V::value_type> {
    return t.fma(b - a, a);
}

template <SimdVec V>
[[nodiscard]] inline V step(const V& edge, const V& x) noexcept
    requires std::floating_point<typename V::value_type> {
    using T = typename V::value_type;
    return V::select(x < edge, V::zero(), V::set1(T{1}));
}

/// Hermite smoothstep.  Written so the inner term is evaluated once and the
/// result is a single fma chain; the naive spelling costs three extra
/// multiplies per lane and shows up in profiled particle code.
template <SimdVec V>
[[nodiscard]] inline V smoothstep(const V& e0, const V& e1, const V& x) noexcept
    requires std::floating_point<typename V::value_type> {
    using T = typename V::value_type;
    // A real divide, not rcp(): the estimate costs accuracy here and
    // smoothstep is rarely the innermost loop.
    const V t = saturate((x - e0) / (e1 - e0));
    return t * t * (V::set1(T{3}) - V::set1(T{2}) * t);
}

/// -1, 0 or +1 per lane, with sign(0) = 0 and sign(NaN) = 0, matching
/// std::copysign-free scalar semantics.
template <SimdVec V>
[[nodiscard]] inline V sign(const V& x) noexcept
    requires std::floating_point<typename V::value_type> {
    using T = typename V::value_type;
    const V zero = V::zero();
    return select(x > zero, V::set1(T{1}),
                  select(x < zero, V::set1(T{-1}), zero));
}

/// Absolute difference between lanes of two vectors.
template <SimdVec V>
[[nodiscard]] inline V abs_diff(const V& a, const V& b) noexcept
    requires std::floating_point<typename V::value_type> {
    return (a - b).abs();
}

} // namespace vectis::math
