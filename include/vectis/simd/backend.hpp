// ===========================================================================
// Vectis - compile-time SIMD engine
// simd/backend.hpp - the primitive layer every vector is built on
// ===========================================================================
//
// A `backend<T, Abi>` wraps ONE hardware register and exposes the operations
// that register supports.  Everything above it - basic_vec, basic_mask, the
// kernels - is written once against this protocol and inherits multi-register
// support for free: a `vec<float, 16>` is one ZMM on AVX-512 and two YMMs on
// AVX2, and no kernel has to know which.
//
// ---------------------------------------------------------------------------
// Protocol.  Every backend<T, Abi> must provide, all static:
//
//   using value_type = T;
//   using reg;                    // one register: backend::lanes lanes of T
//   using mask_reg;               // backend::lanes predicates
//   static constexpr size_t lanes;
//
//   construction
//     reg zero();
//     reg set1(T);
//     reg load(const T* p);       // unaligned-safe
//     void store(T* p, reg);      // unaligned-safe
//     reg iota();                 // 0,1,2,...  (integers only)
//
//   arithmetic      add, sub, mul             (all T)
//                   div                       (floating point, and i32)
//                   fma(a,b,c) = a*b + c      (floating point)
//                   neg, abs
//   floating point  sqrt, rsqrt_approx, rcp_approx
//   ordering        min, max
//   bitwise         band, bor, bxor, bandnot
//   shifts          shl, shr_logical, shr_arith   (integers)
//   comparison      cmpeq, cmpne, cmplt, cmple, cmpgt, cmpge  -> mask_reg
//   selection       select(mask_reg, a, b)    // mask ? a : b
//   masks           mask_true, mask_false, mask_from_bits(bits), mask_bits(m),
//                   mask_test(m, lane), mask_and, mask_or, mask_xor, mask_not
//
// Semantics that MUST match across backends, because the scalar backend is the
// correctness oracle for the vector ones:
//
//   * min(a,b) == (a < b) ? a : b   and   max(a,b) == (a > b) ? a : b
//     This is the hardware rule, NOT std::fmin/std::fmax.  It is asymmetric in
//     the presence of NaN - min(NaN, x) is x, min(x, NaN) is NaN - and the two
//     conventions disagree on signed zero.
//   * Integer shifts are wrapping; a shift count >= bit width is undefined in
//     both C++ and the hardware, so callers must mask it.
//   * Comparisons against NaN are false, exactly as with hardware compare.
//
// ===========================================================================
#pragma once

#include "abi.hpp"

namespace vectis {

/// Primitive operations for one register of `T` under `Abi`.
///
/// Undefined primary template: asking for a (T, Abi) pair that has no backend
/// is a hard error naming the pair, not a silent fallback to something slow.
template <class T, class Abi>
struct backend;

/// True when a backend exists for this element type and ABI.
template <class T, class Abi>
concept HasBackend = requires {
    typename backend<T, Abi>::reg;
    requires backend<T, Abi>::lanes > 0;
};

/// Lanes a backend holds, usable in a context where the backend may not exist.
template <class T, class Abi>
inline constexpr std::size_t backend_lanes_v =
    backend<T, Abi>::lanes;

/// Is `N` a legal lane count for this element type and ABI?
///
/// Any positive N is.  A vector whose lane count is not a multiple of the
/// register width simply rounds up to whole registers and treats the surplus
/// lanes as padding: arithmetic runs on them harmlessly, and only load/store,
/// to_array and operator[] care, each of which handles the tail explicitly.
///
/// This is what makes one source work at every tier.  `f32<8>` is one YMM on
/// AVX2 and half of one ZMM on AVX-512; `f32<6>` is one YMM with two dead lanes
/// on AVX2 and still exactly six observable lanes.  Without this, every
/// hard-coded width would be a portability bug waiting for a wider CPU.
template <class T, class Abi, std::size_t N>
concept ValidLaneCount = HasBackend<T, Abi> && (N > 0);

// --------------------------------------------------------- capability probes
//
// AVX2 has no 64-bit multiply, no 64-bit min/max and no ordered 64-bit
// compare.  A kernel written against it must learn that at compile time rather
// than at the first call, so these concepts are how a kernel says "only where
// the hardware can actually do this".
//
// Constraining on them means a kernel that needs vpmullq simply does not exist
// on AVX2: a compile error naming the call site, rather than a silent scalar
// fallback that quietly runs a quarter speed.

template <class T, class A>
concept BackendHasMul = requires(typename backend<T, A>::reg x) {
    { backend<T, A>::mul(x, x) } -> std::same_as<typename backend<T, A>::reg>;
};
template <class T, class A>
concept BackendHasDiv = requires(typename backend<T, A>::reg x) {
    { backend<T, A>::div(x, x) } -> std::same_as<typename backend<T, A>::reg>;
};
template <class T, class A>
concept BackendHasFma = requires(typename backend<T, A>::reg x) {
    { backend<T, A>::fma(x, x, x) } -> std::same_as<typename backend<T, A>::reg>;
};
template <class T, class A>
concept BackendHasSqrt = requires(typename backend<T, A>::reg x) {
    { backend<T, A>::sqrt(x) } -> std::same_as<typename backend<T, A>::reg>;
};
template <class T, class A>
concept BackendHasMinMax = requires(typename backend<T, A>::reg x) {
    { backend<T, A>::min(x, x) } -> std::same_as<typename backend<T, A>::reg>;
    { backend<T, A>::max(x, x) } -> std::same_as<typename backend<T, A>::reg>;
};
template <class T, class A>
concept BackendHasOrdering = requires(typename backend<T, A>::reg x) {
    { backend<T, A>::cmplt(x, x) } -> std::same_as<typename backend<T, A>::mask_reg>;
};
template <class T, class A>
concept BackendHasShifts = requires(typename backend<T, A>::reg x) {
    { backend<T, A>::shl(x, 1) } -> std::same_as<typename backend<T, A>::reg>;
};
template <class T, class A>
concept BackendHasReciprocal = requires(typename backend<T, A>::reg x) {
    { backend<T, A>::rsqrt_approx(x) } -> std::same_as<typename backend<T, A>::reg>;
    { backend<T, A>::rcp_approx(x) } -> std::same_as<typename backend<T, A>::reg>;
};

} // namespace vectis
