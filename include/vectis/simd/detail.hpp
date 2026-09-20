// ===========================================================================
// Vectis - compile-time SIMD engine
// simd/detail.hpp - small metaprogramming helpers shared by the backends
// ===========================================================================
#pragma once

#include "../core/config.hpp"
#include "../core/concepts.hpp"

#include <bit>
#include <cstdint>
#include <type_traits>

namespace vectis::detail {

/// Unsigned integer of the same width as T, used as the carrier for bitwise
/// operations on floating-point lanes.
template <class T>
struct uint_of;
template <> struct uint_of<float>  { using type = std::uint32_t; };
template <> struct uint_of<double> { using type = std::uint64_t; };
template <class T> requires std::integral<T>
struct uint_of<T> { using type = std::make_unsigned_t<T>; };

template <class T>
using uint_of_t = typename uint_of<T>::type;

/// Reinterpret a scalar as its bit pattern.  Used by the scalar backend to
/// implement band/bor/bxor on floating-point lanes with the same semantics the
/// vector backends get from the hardware.
template <class T>
[[nodiscard]] constexpr uint_of_t<T> to_bits(T v) noexcept {
    return std::bit_cast<uint_of_t<T>>(v);
}

template <class T>
[[nodiscard]] constexpr T from_bits(uint_of_t<T> v) noexcept {
    return std::bit_cast<T>(v);
}

/// Absolute value that is well-defined for every signed type, including the
/// minimum value (where plain `-a` overflows and is UB).
template <std::integral T>
[[nodiscard]] constexpr T abs_int(T a) noexcept {
    using U = std::make_unsigned_t<T>;
    return (a < T{0}) ? static_cast<T>(U{0} - static_cast<U>(a)) : a;
}

template <std::unsigned_integral T>
[[nodiscard]] constexpr T abs_int(T a) noexcept { return a; }

/// Sign-bit-preserving absolute value, written the way the hardware does it:
/// clear the top bit.  Branch-free, and NaN payloads survive untouched.
template <class T>
[[nodiscard]] constexpr T abs_bits(T a) noexcept {
    constexpr auto sign = uint_of_t<T>{1} << (sizeof(T) * 8 - 1);
    return from_bits<T>(static_cast<uint_of_t<T>>(to_bits(a) & ~sign));
}

template <std::floating_point T>
[[nodiscard]] constexpr T neg_bits(T a) noexcept {
    constexpr auto sign = uint_of_t<T>{1} << (sizeof(T) * 8 - 1);
    return from_bits<T>(static_cast<uint_of_t<T>>(to_bits(a) ^ sign));
}

/// Number of lanes a full register holds for this element size.
template <class T, std::size_t RegisterBits>
inline constexpr std::size_t lanes_per_register = RegisterBits / (sizeof(T) * 8);

/// Ceiling division, for computing register counts.
template <std::size_t A, std::size_t B>
inline constexpr std::size_t div_ceil = (A + B - 1) / B;

} // namespace vectis::detail
