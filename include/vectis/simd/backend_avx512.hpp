// ===========================================================================
// Vectis - compile-time SIMD engine
// simd/backend_avx512.hpp - 512-bit backend (AVX-512 F/BW/DQ/VL)
// ===========================================================================
//
// Where AVX2 fakes predicates with all-ones vectors, AVX-512 has eight real
// mask registers.  Three consequences run through this whole file and through
// every kernel built on it:
//
//   * mask_bits() is free.  On AVX2 it is a vmovmskps (a vector-to-GPR move
//     with real latency); here the predicate *already is* a general-purpose
//     integer and reading it costs nothing.  Branchy code - ray-box slab
//     tests, frustum culling - is dramatically better off, not because the
//     arithmetic is 2x wider but because testing 16 lanes at once stopped
//     being expensive.
//   * blend is a mask operation, not a vector one, and blends with an
//     immediate mask are single instructions.
//   * The 64-bit integer story is complete: vpmullq, vpminsq/vpmaxsq and
//     ordered 64-bit compares exist here, so the i64 backend is no longer a
//     subset the way it is on AVX2.
//
// Purely as a comparison of what the two tiers can express:
//   feature                 AVX2        AVX-512
//   predicate storage       vector      k register
//   mask_bits()             vmovmskps   free (move GPR)
//   64-bit multiply         no          vpmullq
//   64-bit min/max/compare  no          yes
//   byte permute across lane no         vpermb (VBMI)
//
// ===========================================================================
#pragma once

#include "backend.hpp"
#include "detail.hpp"

#if defined(VECTIS_USE_AVX512)

namespace vectis::avx512_detail {

/// Broadcast bit i of `bits` into mask lane i.  With real mask registers this
/// is a plain truncation - no vector gymnastics required.
template <class Mask>
[[nodiscard]] inline Mask bits_to_mask(std::uint64_t bits) noexcept {
    return static_cast<Mask>(bits);
}

template <class Mask>
[[nodiscard]] inline Mask all_mask() noexcept {
    return static_cast<Mask>(~std::uint64_t{0});
}

/// Lane index vectors: functions, not macro arguments, because a parenthesised
/// comma list would arrive as a comma *expression*.
[[nodiscard]] inline __m512 iota_ps() noexcept {
    return _mm512_setr_ps(0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f,
                          8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f);
}
[[nodiscard]] inline __m512d iota_pd() noexcept {
    return _mm512_setr_pd(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
}
[[nodiscard]] inline __m512i iota_epi32() noexcept {
    return _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}
[[nodiscard]] inline __m512i set1_epi64(std::int64_t v) noexcept {
    return _mm512_set1_epi64(v);
}
[[nodiscard]] inline __m512i set1_epu64(std::uint64_t v) noexcept {
    return _mm512_set1_epi64(static_cast<long long>(v));
}
[[nodiscard]] inline __m512i abs_epu32(__m512i a) noexcept { return a; }
[[nodiscard]] inline __m512i abs_epu64(__m512i a) noexcept { return a; }

[[nodiscard]] inline __m512i iota_epi64() noexcept {
    return _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
}

[[nodiscard]] inline __m512i loadu_epi(const std::int32_t* p) noexcept {
    return _mm512_loadu_si512(static_cast<const void*>(p));
}
[[nodiscard]] inline __m512i loadu_epi(const std::uint32_t* p) noexcept {
    return _mm512_loadu_si512(static_cast<const void*>(p));
}
[[nodiscard]] inline __m512i loadu_epi(const std::int64_t* p) noexcept {
    return _mm512_loadu_si512(static_cast<const void*>(p));
}
[[nodiscard]] inline __m512i loadu_epi(const std::uint64_t* p) noexcept {
    return _mm512_loadu_si512(static_cast<const void*>(p));
}
inline void storeu_epi(std::int32_t* p, __m512i v) noexcept {
    _mm512_storeu_si512(static_cast<void*>(p), v);
}
inline void storeu_epi(std::uint32_t* p, __m512i v) noexcept {
    _mm512_storeu_si512(static_cast<void*>(p), v);
}
inline void storeu_epi(std::int64_t* p, __m512i v) noexcept {
    _mm512_storeu_si512(static_cast<void*>(p), v);
}
inline void storeu_epi(std::uint64_t* p, __m512i v) noexcept {
    _mm512_storeu_si512(static_cast<void*>(p), v);
}

} // namespace vectis::avx512_detail

namespace vectis {

// ===========================================================================
// Floating point
// ===========================================================================
#define VECTIS_AVX512_FP(TYPE, REG, SUF, MASK, LANES, IOTA)                           \
    template <> struct backend<TYPE, avx512_abi> {                              \
        using value_type = TYPE;                                                \
        using reg        = REG;                                                 \
        using mask_reg   = MASK;                                                \
        static constexpr std::size_t lanes = LANES;                             \
                                                                                \
        [[nodiscard]] static reg zero() noexcept {                              \
            return _mm512_setzero_##SUF();                                      \
        }                                                                       \
        [[nodiscard]] static reg set1(TYPE v) noexcept {                        \
            return _mm512_set1_##SUF(v);                                        \
        }                                                                       \
        [[nodiscard]] static reg load(const TYPE* p) noexcept {                 \
            return _mm512_loadu_##SUF(p);                                       \
        }                                                                       \
        static void store(TYPE* p, reg v) noexcept {                            \
            _mm512_storeu_##SUF(p, v);                                          \
        }                                                                       \
        [[nodiscard]] static reg iota() noexcept {                              \
            return IOTA();                                                      \
        }                                                                       \
                                                                                \
        [[nodiscard]] static reg add(reg a, reg b) noexcept {                   \
            return _mm512_add_##SUF(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg sub(reg a, reg b) noexcept {                   \
            return _mm512_sub_##SUF(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg mul(reg a, reg b) noexcept {                   \
            return _mm512_mul_##SUF(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg div(reg a, reg b) noexcept {                   \
            return _mm512_div_##SUF(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg fma(reg a, reg b, reg c) noexcept {            \
            return _mm512_fmadd_##SUF(a, b, c);                                 \
        }                                                                       \
        [[nodiscard]] static reg neg(reg a) noexcept {                          \
            return _mm512_xor_##SUF(a, _mm512_set1_##SUF(-0.0));                \
        }                                                                       \
        [[nodiscard]] static reg abs(reg a) noexcept {                          \
            return _mm512_andnot_##SUF(_mm512_set1_##SUF(-0.0), a);             \
        }                                                                       \
                                                                                \
        [[nodiscard]] static reg sqrt(reg a) noexcept {                         \
            return _mm512_sqrt_##SUF(a);                                        \
        }                                                                       \
        [[nodiscard]] static reg rsqrt_approx(reg a) noexcept {                 \
            return _mm512_rsqrt14_##SUF(a);                                     \
        }                                                                       \
        [[nodiscard]] static reg rcp_approx(reg a) noexcept {                   \
            return _mm512_rcp14_##SUF(a);                                       \
        }                                                                       \
        [[nodiscard]] static reg min(reg a, reg b) noexcept {                   \
            return _mm512_min_##SUF(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg max(reg a, reg b) noexcept {                   \
            return _mm512_max_##SUF(a, b);                                      \
        }                                                                       \
                                                                                \
        [[nodiscard]] static reg band(reg a, reg b) noexcept {                  \
            return _mm512_and_##SUF(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg bor(reg a, reg b) noexcept {                   \
            return _mm512_or_##SUF(a, b);                                       \
        }                                                                       \
        [[nodiscard]] static reg bxor(reg a, reg b) noexcept {                  \
            return _mm512_xor_##SUF(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg bandnot(reg a, reg b) noexcept {               \
            return _mm512_andnot_##SUF(a, b);                                   \
        }                                                                       \
                                                                                \
        [[nodiscard]] static mask_reg cmpeq(reg a, reg b) noexcept {            \
            return _mm512_cmp_##SUF##_mask(a, b, _CMP_EQ_OQ);                   \
        }                                                                       \
        [[nodiscard]] static mask_reg cmpne(reg a, reg b) noexcept {            \
            return _mm512_cmp_##SUF##_mask(a, b, _CMP_NEQ_UQ);                  \
        }                                                                       \
        [[nodiscard]] static mask_reg cmplt(reg a, reg b) noexcept {            \
            return _mm512_cmp_##SUF##_mask(a, b, _CMP_LT_OQ);                   \
        }                                                                       \
        [[nodiscard]] static mask_reg cmple(reg a, reg b) noexcept {            \
            return _mm512_cmp_##SUF##_mask(a, b, _CMP_LE_OQ);                   \
        }                                                                       \
        [[nodiscard]] static mask_reg cmpgt(reg a, reg b) noexcept {            \
            return _mm512_cmp_##SUF##_mask(a, b, _CMP_GT_OQ);                   \
        }                                                                       \
        [[nodiscard]] static mask_reg cmpge(reg a, reg b) noexcept {            \
            return _mm512_cmp_##SUF##_mask(a, b, _CMP_GE_OQ);                   \
        }                                                                       \
                                                                                \
        /* mask_blend(k, a, b) selects b where k is set, so the operand order   \
           is reversed relative to the name: our select(m, x, y) means          \
           "m ? x : y" and therefore passes (m, y, x). */                       \
        [[nodiscard]] static reg select(mask_reg m, reg a, reg b) noexcept {    \
            return _mm512_mask_blend_##SUF(m, b, a);                            \
        }                                                                       \
                                                                                \
        [[nodiscard]] static mask_reg mask_true() noexcept {                    \
            return avx512_detail::all_mask<MASK>();                             \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_false() noexcept {                   \
            return static_cast<MASK>(0);                                        \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_from_bits(std::uint64_t b) noexcept {\
            return avx512_detail::bits_to_mask<MASK>(b);                        \
        }                                                                       \
        [[nodiscard]] static std::uint64_t mask_bits(mask_reg m) noexcept {     \
            return static_cast<std::uint64_t>(m);                               \
        }                                                                       \
        [[nodiscard]] static bool mask_test(mask_reg m, std::size_t lane) noexcept {\
            return ((static_cast<std::uint64_t>(m) >> lane) & 1u) != 0;         \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_and(mask_reg a, mask_reg b) noexcept {\
            return static_cast<mask_reg>(a & b);                                \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_or(mask_reg a, mask_reg b) noexcept {\
            return static_cast<mask_reg>(a | b);                                \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_xor(mask_reg a, mask_reg b) noexcept {\
            return static_cast<mask_reg>(a ^ b);                                \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_not(mask_reg a) noexcept {           \
            return static_cast<mask_reg>(~a);                                   \
        }                                                                       \
    };

VECTIS_AVX512_FP(float,  __m512,  ps, __mmask16, 16, avx512_detail::iota_ps)
VECTIS_AVX512_FP(double, __m512d, pd, __mmask8,   8, avx512_detail::iota_pd)

#undef VECTIS_AVX512_FP

// ===========================================================================
// 32-bit integers
// ===========================================================================
#define VECTIS_AVX512_I32(TYPE, MASK, MINOP, MAXOP, CMPLT, IOTA, ABSOP)                \
    template <> struct backend<TYPE, avx512_abi> {                              \
        using value_type = TYPE;                                                \
        using reg        = __m512i;                                             \
        using mask_reg   = MASK;                                                \
        static constexpr std::size_t lanes = 16;                                \
                                                                                \
        [[nodiscard]] static reg zero() noexcept {                              \
            return _mm512_setzero_si512();                                      \
        }                                                                       \
        [[nodiscard]] static reg set1(TYPE v) noexcept {                        \
            return _mm512_set1_epi32(static_cast<int>(v));                      \
        }                                                                       \
        [[nodiscard]] static reg load(const TYPE* p) noexcept {                 \
            return avx512_detail::loadu_epi(p);                                 \
        }                                                                       \
        static void store(TYPE* p, reg v) noexcept {                            \
            avx512_detail::storeu_epi(p, v);                                    \
        }                                                                       \
        [[nodiscard]] static reg iota() noexcept {                              \
            return IOTA();                                     \
        }                                                                       \
                                                                                \
        [[nodiscard]] static reg add(reg a, reg b) noexcept {                   \
            return _mm512_add_epi32(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg sub(reg a, reg b) noexcept {                   \
            return _mm512_sub_epi32(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg mul(reg a, reg b) noexcept {                   \
            return _mm512_mullo_epi32(a, b);                                    \
        }                                                                       \
        [[nodiscard]] static reg neg(reg a) noexcept {                          \
            return _mm512_sub_epi32(_mm512_setzero_si512(), a);                 \
        }                                                                       \
        [[nodiscard]] static reg abs(reg a) noexcept {                          \
            return ABSOP(a);                                                    \
        }                                                                       \
        [[nodiscard]] static reg min(reg a, reg b) noexcept { return MINOP(a, b); } \
        [[nodiscard]] static reg max(reg a, reg b) noexcept { return MAXOP(a, b); } \
                                                                                \
        [[nodiscard]] static reg band(reg a, reg b) noexcept {                  \
            return _mm512_and_si512(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg bor(reg a, reg b) noexcept {                   \
            return _mm512_or_si512(a, b);                                       \
        }                                                                       \
        [[nodiscard]] static reg bxor(reg a, reg b) noexcept {                  \
            return _mm512_xor_si512(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg bandnot(reg a, reg b) noexcept {               \
            return _mm512_andnot_si512(a, b);                                   \
        }                                                                       \
        [[nodiscard]] static reg shl(reg a, int n) noexcept {                   \
            return _mm512_sllv_epi32(a, _mm512_set1_epi32(n));                  \
        }                                                                       \
        [[nodiscard]] static reg shr_logical(reg a, int n) noexcept {           \
            return _mm512_srlv_epi32(a, _mm512_set1_epi32(n));                  \
        }                                                                       \
        [[nodiscard]] static reg shr_arith(reg a, int n) noexcept {             \
            return _mm512_srav_epi32(a, _mm512_set1_epi32(n));                  \
        }                                                                       \
                                                                                \
        [[nodiscard]] static mask_reg cmpeq(reg a, reg b) noexcept {            \
            return _mm512_cmpeq_epi32_mask(a, b);                               \
        }                                                                       \
        [[nodiscard]] static mask_reg cmpne(reg a, reg b) noexcept {            \
            return _mm512_cmpneq_epi32_mask(a, b);                              \
        }                                                                       \
        [[nodiscard]] static mask_reg cmplt(reg a, reg b) noexcept {            \
            return CMPLT(a, b);                                                 \
        }                                                                       \
        [[nodiscard]] static mask_reg cmple(reg a, reg b) noexcept {            \
            return mask_not(CMPLT(b, a));                                       \
        }                                                                       \
        [[nodiscard]] static mask_reg cmpgt(reg a, reg b) noexcept {            \
            return CMPLT(b, a);                                                 \
        }                                                                       \
        [[nodiscard]] static mask_reg cmpge(reg a, reg b) noexcept {            \
            return mask_not(CMPLT(a, b));                                       \
        }                                                                       \
                                                                                \
        [[nodiscard]] static reg select(mask_reg m, reg a, reg b) noexcept {    \
            return _mm512_mask_blend_epi32(m, b, a);                            \
        }                                                                       \
                                                                                \
        [[nodiscard]] static mask_reg mask_true() noexcept {                    \
            return avx512_detail::all_mask<MASK>();                             \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_false() noexcept {                   \
            return static_cast<MASK>(0);                                        \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_from_bits(std::uint64_t b) noexcept {\
            return avx512_detail::bits_to_mask<MASK>(b);                        \
        }                                                                       \
        [[nodiscard]] static std::uint64_t mask_bits(mask_reg m) noexcept {     \
            return static_cast<std::uint64_t>(m);                               \
        }                                                                       \
        [[nodiscard]] static bool mask_test(mask_reg m, std::size_t lane) noexcept {\
            return ((static_cast<std::uint64_t>(m) >> lane) & 1u) != 0;         \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_and(mask_reg a, mask_reg b) noexcept {\
            return static_cast<mask_reg>(a & b);                                \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_or(mask_reg a, mask_reg b) noexcept {\
            return static_cast<mask_reg>(a | b);                                \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_xor(mask_reg a, mask_reg b) noexcept {\
            return static_cast<mask_reg>(a ^ b);                                \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_not(mask_reg a) noexcept {           \
            return static_cast<mask_reg>(~a);                                   \
        }                                                                       \
    };

VECTIS_AVX512_I32(std::int32_t, __mmask16,
                  _mm512_min_epi32, _mm512_max_epi32,
                  _mm512_cmplt_epi32_mask,
                  avx512_detail::iota_epi32,
                  _mm512_abs_epi32)
VECTIS_AVX512_I32(std::uint32_t, __mmask16,
                  _mm512_min_epu32, _mm512_max_epu32,
                  _mm512_cmplt_epu32_mask,
                  avx512_detail::iota_epi32,
                  avx512_detail::abs_epu32)

#undef VECTIS_AVX512_I32

// ===========================================================================
// 64-bit integers.  Complete here, unlike AVX2: vpmullq, vpminsq/vpmaxsq and
// ordered 64-bit compares all exist, so this backend is a superset of the
// AVX2 one rather than a subset.
// ===========================================================================
#define VECTIS_AVX512_I64(TYPE, MASK, SET1, MINOP, MAXOP, CMPLT, ABSOP, IOTA)           \
    template <> struct backend<TYPE, avx512_abi> {                              \
        using value_type = TYPE;                                                \
        using reg        = __m512i;                                             \
        using mask_reg   = MASK;                                                \
        static constexpr std::size_t lanes = 8;                                 \
                                                                                \
        [[nodiscard]] static reg zero() noexcept {                              \
            return _mm512_setzero_si512();                                      \
        }                                                                       \
        [[nodiscard]] static reg set1(TYPE v) noexcept { return SET1(v); }      \
        [[nodiscard]] static reg load(const TYPE* p) noexcept {                 \
            return avx512_detail::loadu_epi(p);                                 \
        }                                                                       \
        static void store(TYPE* p, reg v) noexcept {                            \
            avx512_detail::storeu_epi(p, v);                                    \
        }                                                                       \
        [[nodiscard]] static reg iota() noexcept {                              \
            return IOTA();                  \
        }                                                                       \
                                                                                \
        [[nodiscard]] static reg add(reg a, reg b) noexcept {                   \
            return _mm512_add_epi64(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg sub(reg a, reg b) noexcept {                   \
            return _mm512_sub_epi64(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg mul(reg a, reg b) noexcept {                   \
            return _mm512_mullo_epi64(a, b);                                    \
        }                                                                       \
        [[nodiscard]] static reg neg(reg a) noexcept {                          \
            return _mm512_sub_epi64(_mm512_setzero_si512(), a);                 \
        }                                                                       \
        [[nodiscard]] static reg abs(reg a) noexcept { return ABSOP(a); }       \
        [[nodiscard]] static reg min(reg a, reg b) noexcept { return MINOP(a, b); } \
        [[nodiscard]] static reg max(reg a, reg b) noexcept { return MAXOP(a, b); } \
                                                                                \
        [[nodiscard]] static reg band(reg a, reg b) noexcept {                  \
            return _mm512_and_si512(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg bor(reg a, reg b) noexcept {                   \
            return _mm512_or_si512(a, b);                                       \
        }                                                                       \
        [[nodiscard]] static reg bxor(reg a, reg b) noexcept {                  \
            return _mm512_xor_si512(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg bandnot(reg a, reg b) noexcept {               \
            return _mm512_andnot_si512(a, b);                                   \
        }                                                                       \
        [[nodiscard]] static reg shl(reg a, int n) noexcept {                   \
            return _mm512_sllv_epi64(a, _mm512_set1_epi64(n));                 \
        }                                                                       \
        [[nodiscard]] static reg shr_logical(reg a, int n) noexcept {           \
            return _mm512_srlv_epi64(a, _mm512_set1_epi64(n));                 \
        }                                                                       \
        [[nodiscard]] static reg shr_arith(reg a, int n) noexcept {             \
            return _mm512_srav_epi64(a, _mm512_set1_epi64(n));                 \
        }                                                                       \
                                                                                \
        [[nodiscard]] static mask_reg cmpeq(reg a, reg b) noexcept {            \
            return _mm512_cmpeq_epi64_mask(a, b);                               \
        }                                                                       \
        [[nodiscard]] static mask_reg cmpne(reg a, reg b) noexcept {            \
            return _mm512_cmpneq_epi64_mask(a, b);                              \
        }                                                                       \
        [[nodiscard]] static mask_reg cmplt(reg a, reg b) noexcept {            \
            return CMPLT(a, b);                                                 \
        }                                                                       \
        [[nodiscard]] static mask_reg cmple(reg a, reg b) noexcept {            \
            return mask_not(CMPLT(b, a));                                       \
        }                                                                       \
        [[nodiscard]] static mask_reg cmpgt(reg a, reg b) noexcept {            \
            return CMPLT(b, a);                                                 \
        }                                                                       \
        [[nodiscard]] static mask_reg cmpge(reg a, reg b) noexcept {            \
            return mask_not(CMPLT(a, b));                                       \
        }                                                                       \
                                                                                \
        [[nodiscard]] static reg select(mask_reg m, reg a, reg b) noexcept {    \
            return _mm512_mask_blend_epi64(m, b, a);                            \
        }                                                                       \
                                                                                \
        [[nodiscard]] static mask_reg mask_true() noexcept {                    \
            return avx512_detail::all_mask<MASK>();                             \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_false() noexcept {                   \
            return static_cast<MASK>(0);                                        \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_from_bits(std::uint64_t b) noexcept {\
            return avx512_detail::bits_to_mask<MASK>(b);                        \
        }                                                                       \
        [[nodiscard]] static std::uint64_t mask_bits(mask_reg m) noexcept {     \
            return static_cast<std::uint64_t>(m);                               \
        }                                                                       \
        [[nodiscard]] static bool mask_test(mask_reg m, std::size_t lane) noexcept {\
            return ((static_cast<std::uint64_t>(m) >> lane) & 1u) != 0;         \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_and(mask_reg a, mask_reg b) noexcept {\
            return static_cast<mask_reg>(a & b);                                \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_or(mask_reg a, mask_reg b) noexcept {\
            return static_cast<mask_reg>(a | b);                                \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_xor(mask_reg a, mask_reg b) noexcept {\
            return static_cast<mask_reg>(a ^ b);                                \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_not(mask_reg a) noexcept {           \
            return static_cast<mask_reg>(~a);                                   \
        }                                                                       \
    };

VECTIS_AVX512_I64(std::int64_t, __mmask8,
                  avx512_detail::set1_epi64,
                  _mm512_min_epi64, _mm512_max_epi64,
                  _mm512_cmplt_epi64_mask,
                  _mm512_abs_epi64,
                  avx512_detail::iota_epi64)
VECTIS_AVX512_I64(std::uint64_t, __mmask8,
                  avx512_detail::set1_epu64,
                  _mm512_min_epu64, _mm512_max_epu64,
                  _mm512_cmplt_epu64_mask,
                  avx512_detail::abs_epu64,
                  avx512_detail::iota_epi64)

#undef VECTIS_AVX512_I64

} // namespace vectis

#endif // VECTIS_USE_AVX512
