// ===========================================================================
// Vectis - compile-time SIMD engine
// simd/backend_scalar.hpp - the reference backend
// ===========================================================================
//
// One lane per "register", implemented with ordinary C++ operators.  This is
// not dead weight: it is the correctness oracle every vector backend is tested
// against, and it is what lets the library run on aarch64, WASM or any future
// target without a vector backend existing yet.
//
// The semantics here deliberately mirror hardware, not <cmath>:
//   min(a,b) = (a < b) ? a : b    - not std::fmin, which differs on NaN and
//                                   on the sign of zero
//   abs(x)   = clear the sign bit  - so NaN payloads and -0.0 behave like the
//                                   vector path
//
// ===========================================================================
#pragma once

#include "backend.hpp"
#include "detail.hpp"

#include <cmath>
#include <cstdint>

namespace vectis {

template <Vectorizable T>
struct backend<T, scalar_abi> {
    using value_type = T;
    using reg        = T;
    using mask_reg   = bool;

    static constexpr std::size_t lanes = 1;

    // ------------------------------------------------------------ construction
    [[nodiscard]] static constexpr reg zero() noexcept { return T{0}; }
    [[nodiscard]] static constexpr reg set1(T v) noexcept { return v; }
    [[nodiscard]] static reg load(const T* p) noexcept { return *p; }
    static void store(T* p, reg v) noexcept { *p = v; }
    /// Lane index of the only lane: always 0.  Present for every T so that
    /// basic_vec::iota works uniformly, including float ramps.
    [[nodiscard]] static constexpr reg iota() noexcept { return T{0}; }

    // -------------------------------------------------------------- arithmetic
    [[nodiscard]] static constexpr reg add(reg a, reg b) noexcept {
        return static_cast<T>(a + b);
    }
    [[nodiscard]] static constexpr reg sub(reg a, reg b) noexcept {
        return static_cast<T>(a - b);
    }
    [[nodiscard]] static constexpr reg mul(reg a, reg b) noexcept {
        return static_cast<T>(a * b);
    }
    [[nodiscard]] static constexpr reg div(reg a, reg b) noexcept
        requires std::floating_point<T> {
        return a / b;
    }
    [[nodiscard]] static constexpr reg fma(reg a, reg b, reg c) noexcept
        requires std::floating_point<T> {
        return std::fma(a, b, c);
    }
    [[nodiscard]] static constexpr reg neg(reg a) noexcept {
        return static_cast<T>(-a);
    }
    [[nodiscard]] static constexpr reg abs(reg a) noexcept {
        if constexpr (std::floating_point<T>) return detail::abs_bits(a);
        else return detail::abs_int(a);
    }

    // --------------------------------------------------------- floating point
    [[nodiscard]] static reg sqrt(reg a) noexcept requires std::floating_point<T> {
        return std::sqrt(a);
    }
    /// Raw hardware reciprocal-square-root quality.  The scalar path cannot do
    /// better than exact, which is *more* accurate than any vector estimate -
    /// so tests of the approximate form must use a tolerance, and the refined
    /// form (vectis::math::rsqrt) is the one to compare against.
    [[nodiscard]] static reg rsqrt_approx(reg a) noexcept requires std::floating_point<T> {
        return T{1} / std::sqrt(a);
    }
    [[nodiscard]] static reg rcp_approx(reg a) noexcept requires std::floating_point<T> {
        return T{1} / a;
    }

    // ---------------------------------------------------------------- ordering
    [[nodiscard]] static constexpr reg min(reg a, reg b) noexcept {
        return (a < b) ? a : b;
    }
    [[nodiscard]] static constexpr reg max(reg a, reg b) noexcept {
        return (a > b) ? a : b;
    }

    // ----------------------------------------------------------------- bitwise
    [[nodiscard]] static constexpr reg band(reg a, reg b) noexcept {
        return detail::from_bits<T>(static_cast<detail::uint_of_t<T>>(
            detail::to_bits(a) & detail::to_bits(b)));
    }
    [[nodiscard]] static constexpr reg bor(reg a, reg b) noexcept {
        return detail::from_bits<T>(static_cast<detail::uint_of_t<T>>(
            detail::to_bits(a) | detail::to_bits(b)));
    }
    [[nodiscard]] static constexpr reg bxor(reg a, reg b) noexcept {
        return detail::from_bits<T>(static_cast<detail::uint_of_t<T>>(
            detail::to_bits(a) ^ detail::to_bits(b)));
    }
    /// (~a) & b - Intel's andnot operand order, kept for consistency with the
    /// vector backends so a kernel reads the same at every tier.
    [[nodiscard]] static constexpr reg bandnot(reg a, reg b) noexcept {
        return detail::from_bits<T>(static_cast<detail::uint_of_t<T>>(
            static_cast<detail::uint_of_t<T>>(~detail::to_bits(a)) &
            detail::to_bits(b)));
    }

    // ------------------------------------------------------------------ shifts
    [[nodiscard]] static constexpr reg shl(reg a, int n) noexcept
        requires std::integral<T> {
        using U = std::make_unsigned_t<T>;
        return static_cast<T>(static_cast<U>(a) << n);
    }
    [[nodiscard]] static constexpr reg shr_logical(reg a, int n) noexcept
        requires std::integral<T> {
        using U = std::make_unsigned_t<T>;
        return static_cast<T>(static_cast<U>(a) >> n);
    }
    [[nodiscard]] static constexpr reg shr_arith(reg a, int n) noexcept
        requires std::signed_integral<T> {
        return static_cast<T>(a >> n);
    }

    // ------------------------------------------------------------- comparisons
    [[nodiscard]] static constexpr mask_reg cmpeq(reg a, reg b) noexcept { return a == b; }
    [[nodiscard]] static constexpr mask_reg cmpne(reg a, reg b) noexcept { return a != b; }
    [[nodiscard]] static constexpr mask_reg cmplt(reg a, reg b) noexcept { return a <  b; }
    [[nodiscard]] static constexpr mask_reg cmple(reg a, reg b) noexcept { return a <= b; }
    [[nodiscard]] static constexpr mask_reg cmpgt(reg a, reg b) noexcept { return a >  b; }
    [[nodiscard]] static constexpr mask_reg cmpge(reg a, reg b) noexcept { return a >= b; }

    // --------------------------------------------------------------- selection
    [[nodiscard]] static constexpr reg select(mask_reg m, reg a, reg b) noexcept {
        return m ? a : b;
    }

    // ------------------------------------------------------------------- masks
    [[nodiscard]] static constexpr mask_reg mask_true() noexcept { return true; }
    [[nodiscard]] static constexpr mask_reg mask_false() noexcept { return false; }
    [[nodiscard]] static constexpr mask_reg mask_from_bits(std::uint64_t b) noexcept {
        return (b & 1u) != 0;
    }
    [[nodiscard]] static constexpr std::uint64_t mask_bits(mask_reg m) noexcept {
        return m ? 1u : 0u;
    }
    [[nodiscard]] static constexpr bool mask_test(mask_reg m, std::size_t lane) noexcept {
        return lane == 0 && m;
    }
    [[nodiscard]] static constexpr mask_reg mask_and(mask_reg a, mask_reg b) noexcept { return a && b; }
    [[nodiscard]] static constexpr mask_reg mask_or (mask_reg a, mask_reg b) noexcept { return a || b; }
    [[nodiscard]] static constexpr mask_reg mask_xor(mask_reg a, mask_reg b) noexcept { return a != b; }
    [[nodiscard]] static constexpr mask_reg mask_not(mask_reg a) noexcept { return !a; }
};

} // namespace vectis
