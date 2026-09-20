// ===========================================================================
// Vectis - compile-time SIMD engine
// simd/backend_avx2.hpp - 256-bit backend (AVX2 + FMA)
// ===========================================================================
//
// The interesting constraint here is that AVX2 is *not* a scaled-down AVX-512.
// The differences that leak into this file, and that kernels must respect:
//
//   * No mask registers.  A predicate is an all-ones/all-zeros vector, so
//     select() is a blend and mask_bits() costs a vmovmskps.
//   * No 64-bit integer multiply, no 64-bit min/max, no 64-bit arithmetic
//     shift, and no 64-bit lane compare other than cmpeq.  The i64 backend
//     therefore offers a subset only, and asking it for something it lacks is
//     a compile error - never a silent scalar fallback.
//   * No reciprocal estimate for double, so rsqrt_approx/rcp_approx on double
//     return the correctly rounded answer.  That is better than an estimate,
//     and the Newton refinement in vectis::math::rsqrt converges either way.
//
// ===========================================================================
#pragma once

#include "backend.hpp"
#include "detail.hpp"

#if defined(VECTIS_USE_AVX2)

namespace vectis::avx2_detail {

/// All-ones in every bit: the AVX2 spelling of "true".
[[nodiscard]] inline __m256i ones_epi32() noexcept { return _mm256_set1_epi32(-1); }

/// Broadcast bit i of `bits` to every bit of 32-bit lane i.
[[nodiscard]] inline __m256i bits_to_mask_epi32(std::uint64_t bits) noexcept {
    const __m256i one    = _mm256_set1_epi32(1);
    const __m256i bitpos = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i spread =
        _mm256_sllv_epi32(_mm256_set1_epi32(static_cast<int>(bits)), bitpos);
    return _mm256_cmpeq_epi32(_mm256_and_si256(spread, one), one);
}

/// Broadcast bit i of `bits` to every bit of 64-bit lane i.
[[nodiscard]] inline __m256i bits_to_mask_epi64(std::uint64_t bits) noexcept {
    const __m256i one    = _mm256_set1_epi64x(1);
    const __m256i bitpos = _mm256_setr_epi64x(0, 1, 2, 3);
    const __m256i spread =
        _mm256_sllv_epi64(_mm256_set1_epi64x(static_cast<long long>(bits)), bitpos);
    return _mm256_cmpeq_epi64(_mm256_and_si256(spread, one), one);
}

/// All-ones per 64-bit lane where that lane is negative.  Emulates the absent
/// _mm256_srai_epi64 by moving each qword's sign bit into both of its dwords.
[[nodiscard]] inline __m256i sign_mask_epi64(__m256i a) noexcept {
    const __m256i high_dwords = _mm256_shuffle_epi32(a, _MM_SHUFFLE(3, 3, 1, 1));
    return _mm256_srai_epi32(high_dwords, 31);
}

/// Arithmetic shift right by a runtime amount, for 64-bit lanes.
[[nodiscard]] inline __m256i sra_epi64(__m256i a, int n) noexcept {
    if (n == 0) return a;
    const __m256i cnt     = _mm256_set1_epi64x(n);
    const __m256i negcnt  = _mm256_set1_epi64x(64 - n);
    const __m256i logical = _mm256_srlv_epi64(a, cnt);
    const __m256i fill    = _mm256_sllv_epi64(sign_mask_epi64(a), negcnt);
    return _mm256_or_si256(logical, fill);
}

/// abs() on an unsigned lane is the identity: |x| == x for x >= 0, and an
/// unsigned lane is never negative.  Matching the scalar oracle here matters
/// more than matching what the signed instruction would do.
[[nodiscard]] inline __m256i abs_epu32(__m256i a) noexcept { return a; }
[[nodiscard]] inline __m256i abs_epu64(__m256i a) noexcept { return a; }

/// Absolute value for signed 64-bit lanes.  There is no vpabsq in AVX2, but the
/// two's-complement identity (a ^ s) - s with s = a >> 63 needs only ops AVX2
/// has.  INT64_MIN maps to itself, exactly as vpabsq and the scalar oracle do.
[[nodiscard]] inline __m256i abs_epi64(__m256i a) noexcept {
    const __m256i sign = sra_epi64(a, 63);
    return _mm256_sub_epi64(_mm256_xor_si256(a, sign), sign);
}

/// Broadcast helpers for 64-bit lanes.  Named functions rather than lambdas so
/// that a parameter named `v` cannot shadow the `set1(TYPE v)` it is expanded
/// into.
[[nodiscard]] inline __m256i set1_epi64(std::int64_t v) noexcept {
    return _mm256_set1_epi64x(v);
}
[[nodiscard]] inline __m256i set1_epu64(std::uint64_t v) noexcept {
    return _mm256_set1_epi64x(static_cast<long long>(v));
}

/// Sign-flip constant for turning a signed compare into an unsigned one.
[[nodiscard]] inline __m256i signbit_epi32() noexcept {
    return _mm256_set1_epi32(static_cast<int>(0x80000000u));
}

// Register-typed helpers.  Overloading on the register type is what lets one
// macro below cover every element type without a token-pasting zoo.
[[nodiscard]] inline __m256  rsqrt_estimate(__m256 a)  noexcept { return _mm256_rsqrt_ps(a); }
[[nodiscard]] inline __m256  rcp_estimate(__m256 a)    noexcept { return _mm256_rcp_ps(a); }
[[nodiscard]] inline __m256d rsqrt_estimate(__m256d a) noexcept {
    return _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(a));
}
[[nodiscard]] inline __m256d rcp_estimate(__m256d a) noexcept {
    return _mm256_div_pd(_mm256_set1_pd(1.0), a);
}

/// Mask construction, keyed on the *element* type rather than the register
/// type.  GCC ignores the `may_alias` attribute on vector types used as
/// template arguments, so keying on __m256 would emit a -Wignored-attributes
/// warning at every use site; keying on float does not, and reads better.
template <class T> struct traits;
template <> struct traits<float> {
    using reg = __m256;
    [[nodiscard]] static reg from_bits(std::uint64_t b) noexcept {
        return _mm256_castsi256_ps(bits_to_mask_epi32(b));
    }
    [[nodiscard]] static reg all() noexcept {
        return _mm256_castsi256_ps(ones_epi32());
    }
    [[nodiscard]] static std::uint64_t bits(reg m) noexcept {
        return static_cast<std::uint64_t>(_mm256_movemask_ps(m));
    }
    [[nodiscard]] static reg iota() noexcept {
        return _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
    }
};
template <> struct traits<double> {
    using reg = __m256d;
    [[nodiscard]] static reg from_bits(std::uint64_t b) noexcept {
        return _mm256_castsi256_pd(bits_to_mask_epi64(b));
    }
    [[nodiscard]] static reg all() noexcept {
        return _mm256_castsi256_pd(ones_epi32());
    }
    [[nodiscard]] static std::uint64_t bits(reg m) noexcept {
        return static_cast<std::uint64_t>(_mm256_movemask_pd(m));
    }
    [[nodiscard]] static reg iota() noexcept {
        return _mm256_setr_pd(0, 1, 2, 3);
    }
};
/// 32-bit integer lanes: the mask rides in the vector, read out with movemask
/// on the corresponding float view.
template <class T> struct traits_i32 {
    using reg = __m256i;
    [[nodiscard]] static reg from_bits(std::uint64_t b) noexcept {
        return bits_to_mask_epi32(b);
    }
    [[nodiscard]] static reg all() noexcept { return ones_epi32(); }
    [[nodiscard]] static std::uint64_t bits(reg m) noexcept {
        return static_cast<std::uint64_t>(_mm256_movemask_ps(_mm256_castsi256_ps(m)));
    }
    [[nodiscard]] static reg iota() noexcept {
        return _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    }
};
template <> struct traits<std::int32_t>  : traits_i32<std::int32_t> {};
template <> struct traits<std::uint32_t> : traits_i32<std::uint32_t> {};

/// 64-bit integer lanes: only 4 predicates, so movemask_pd reads them out.
template <class T> struct traits_i64 {
    using reg = __m256i;
    [[nodiscard]] static reg from_bits(std::uint64_t b) noexcept {
        return bits_to_mask_epi64(b);
    }
    [[nodiscard]] static reg all() noexcept { return ones_epi32(); }
    [[nodiscard]] static std::uint64_t bits(reg m) noexcept {
        return static_cast<std::uint64_t>(_mm256_movemask_pd(_mm256_castsi256_pd(m)));
    }
    [[nodiscard]] static reg iota() noexcept {
        return _mm256_setr_epi64x(0, 1, 2, 3);
    }
};
template <> struct traits<std::int64_t>  : traits_i64<std::int64_t> {};
template <> struct traits<std::uint64_t> : traits_i64<std::uint64_t> {};

/// Unaligned integer loads and stores.  GCC's intrinsics take the `_u` pointer
/// types, which is why these are wrapped rather than open-coded.
[[nodiscard]] inline __m256i loadu_epi(const std::int32_t* p) noexcept {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i_u*>(p));
}
[[nodiscard]] inline __m256i loadu_epi(const std::uint32_t* p) noexcept {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i_u*>(p));
}
[[nodiscard]] inline __m256i loadu_epi(const std::int64_t* p) noexcept {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i_u*>(p));
}
[[nodiscard]] inline __m256i loadu_epi(const std::uint64_t* p) noexcept {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i_u*>(p));
}
inline void storeu_epi(std::int32_t* p, __m256i v) noexcept {
    _mm256_storeu_si256(reinterpret_cast<__m256i_u*>(p), v);
}
inline void storeu_epi(std::uint32_t* p, __m256i v) noexcept {
    _mm256_storeu_si256(reinterpret_cast<__m256i_u*>(p), v);
}
inline void storeu_epi(std::int64_t* p, __m256i v) noexcept {
    _mm256_storeu_si256(reinterpret_cast<__m256i_u*>(p), v);
}
inline void storeu_epi(std::uint64_t* p, __m256i v) noexcept {
    _mm256_storeu_si256(reinterpret_cast<__m256i_u*>(p), v);
}

} // namespace vectis::avx2_detail

namespace vectis {

// ===========================================================================
// Floating point.  float and double differ only in their intrinsic suffix, so
// one macro covers both without hiding anything type-specific.
// ===========================================================================
#define VECTIS_AVX2_FP(TYPE, REG, SUF)                                          \
    template <> struct backend<TYPE, avx2_abi> {                                \
        using value_type = TYPE;                                                \
        using reg        = REG;                                                 \
        using mask_reg   = REG;                                                 \
        static constexpr std::size_t lanes = 32 / sizeof(TYPE);                 \
                                                                                \
        [[nodiscard]] static reg zero() noexcept {                              \
            return _mm256_setzero_##SUF();                                      \
        }                                                                       \
        [[nodiscard]] static reg set1(TYPE v) noexcept {                        \
            return _mm256_set1_##SUF(v);                                        \
        }                                                                       \
        [[nodiscard]] static reg load(const TYPE* p) noexcept {                 \
            return _mm256_loadu_##SUF(p);                                       \
        }                                                                       \
        static void store(TYPE* p, reg v) noexcept {                            \
            _mm256_storeu_##SUF(p, v);                                          \
        }                                                                       \
        [[nodiscard]] static reg iota() noexcept {                              \
            return avx2_detail::traits<TYPE>::iota();                           \
        }                                                                       \
                                                                                \
        [[nodiscard]] static reg add(reg a, reg b) noexcept {                   \
            return _mm256_add_##SUF(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg sub(reg a, reg b) noexcept {                   \
            return _mm256_sub_##SUF(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg mul(reg a, reg b) noexcept {                   \
            return _mm256_mul_##SUF(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg div(reg a, reg b) noexcept {                   \
            return _mm256_div_##SUF(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg fma(reg a, reg b, reg c) noexcept {            \
            return _mm256_fmadd_##SUF(a, b, c);                                 \
        }                                                                       \
        /* Negate by flipping the sign lane: -0.0 is exactly the sign bit set.  \
           Branch-free, and it commutes with NaN the way the hardware does. */  \
        [[nodiscard]] static reg neg(reg a) noexcept {                          \
            return _mm256_xor_##SUF(a, _mm256_set1_##SUF(-0.0));                \
        }                                                                       \
        [[nodiscard]] static reg abs(reg a) noexcept {                          \
            return _mm256_andnot_##SUF(_mm256_set1_##SUF(-0.0), a);             \
        }                                                                       \
                                                                                \
        [[nodiscard]] static reg sqrt(reg a) noexcept {                         \
            return _mm256_sqrt_##SUF(a);                                        \
        }                                                                       \
        [[nodiscard]] static reg rsqrt_approx(reg a) noexcept {                 \
            return avx2_detail::rsqrt_estimate(a);                              \
        }                                                                       \
        [[nodiscard]] static reg rcp_approx(reg a) noexcept {                   \
            return avx2_detail::rcp_estimate(a);                                \
        }                                                                       \
        [[nodiscard]] static reg min(reg a, reg b) noexcept {                   \
            return _mm256_min_##SUF(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg max(reg a, reg b) noexcept {                   \
            return _mm256_max_##SUF(a, b);                                      \
        }                                                                       \
                                                                                \
        [[nodiscard]] static reg band(reg a, reg b) noexcept {                  \
            return _mm256_and_##SUF(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg bor(reg a, reg b) noexcept {                   \
            return _mm256_or_##SUF(a, b);                                       \
        }                                                                       \
        [[nodiscard]] static reg bxor(reg a, reg b) noexcept {                  \
            return _mm256_xor_##SUF(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg bandnot(reg a, reg b) noexcept {               \
            return _mm256_andnot_##SUF(a, b);                                   \
        }                                                                       \
                                                                                \
        [[nodiscard]] static mask_reg cmpeq(reg a, reg b) noexcept {            \
            return _mm256_cmp_##SUF(a, b, _CMP_EQ_OQ);                          \
        }                                                                       \
        [[nodiscard]] static mask_reg cmpne(reg a, reg b) noexcept {            \
            return _mm256_cmp_##SUF(a, b, _CMP_NEQ_UQ);                         \
        }                                                                       \
        [[nodiscard]] static mask_reg cmplt(reg a, reg b) noexcept {            \
            return _mm256_cmp_##SUF(a, b, _CMP_LT_OQ);                          \
        }                                                                       \
        [[nodiscard]] static mask_reg cmple(reg a, reg b) noexcept {            \
            return _mm256_cmp_##SUF(a, b, _CMP_LE_OQ);                          \
        }                                                                       \
        [[nodiscard]] static mask_reg cmpgt(reg a, reg b) noexcept {            \
            return _mm256_cmp_##SUF(a, b, _CMP_GT_OQ);                          \
        }                                                                       \
        [[nodiscard]] static mask_reg cmpge(reg a, reg b) noexcept {            \
            return _mm256_cmp_##SUF(a, b, _CMP_GE_OQ);                          \
        }                                                                       \
                                                                                \
        [[nodiscard]] static reg select(mask_reg m, reg a, reg b) noexcept {    \
            return _mm256_blendv_##SUF(b, a, m);                                \
        }                                                                       \
                                                                                \
        [[nodiscard]] static mask_reg mask_true() noexcept {                    \
            return avx2_detail::traits<TYPE>::all();                       \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_false() noexcept {                   \
            return _mm256_setzero_##SUF();                                      \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_from_bits(std::uint64_t b) noexcept {\
            return avx2_detail::traits<TYPE>::from_bits(b);                \
        }                                                                       \
        [[nodiscard]] static std::uint64_t mask_bits(mask_reg m) noexcept {     \
            return avx2_detail::traits<TYPE>::bits(m);                     \
        }                                                                       \
        [[nodiscard]] static bool mask_test(mask_reg m, std::size_t lane) noexcept {\
            return ((mask_bits(m) >> lane) & 1u) != 0;                          \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_and(mask_reg a, mask_reg b) noexcept {\
            return _mm256_and_##SUF(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_or(mask_reg a, mask_reg b) noexcept {\
            return _mm256_or_##SUF(a, b);                                       \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_xor(mask_reg a, mask_reg b) noexcept {\
            return _mm256_xor_##SUF(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_not(mask_reg a) noexcept {           \
            return _mm256_xor_##SUF(a, mask_true());                            \
        }                                                                       \
    };

VECTIS_AVX2_FP(float, __m256, ps)
VECTIS_AVX2_FP(double, __m256d, pd)

#undef VECTIS_AVX2_FP

// ===========================================================================
// 32-bit integers.  Signed and unsigned share every operation except min/max
// and the ordering compares, which need the sign-flip trick for unsigned.
// ===========================================================================
#define VECTIS_AVX2_I32(TYPE, MINOP, MAXOP, CMPGT, ABSOP)                        \
    template <> struct backend<TYPE, avx2_abi> {                                \
        using value_type = TYPE;                                                \
        using reg        = __m256i;                                             \
        using mask_reg   = __m256i;                                             \
        static constexpr std::size_t lanes = 8;                                 \
                                                                                \
        [[nodiscard]] static reg zero() noexcept {                              \
            return _mm256_setzero_si256();                                      \
        }                                                                       \
        [[nodiscard]] static reg set1(TYPE v) noexcept {                        \
            return _mm256_set1_epi32(static_cast<int>(v));                      \
        }                                                                       \
        [[nodiscard]] static reg load(const TYPE* p) noexcept {                 \
            return avx2_detail::loadu_epi(p);                                   \
        }                                                                       \
        static void store(TYPE* p, reg v) noexcept {                            \
            avx2_detail::storeu_epi(p, v);                                      \
        }                                                                       \
        [[nodiscard]] static reg iota() noexcept {                              \
            return avx2_detail::traits<TYPE>::iota();                           \
        }                                                                       \
                                                                                \
        [[nodiscard]] static reg add(reg a, reg b) noexcept {                   \
            return _mm256_add_epi32(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg sub(reg a, reg b) noexcept {                   \
            return _mm256_sub_epi32(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg mul(reg a, reg b) noexcept {                   \
            return _mm256_mullo_epi32(a, b);                                    \
        }                                                                       \
        [[nodiscard]] static reg neg(reg a) noexcept {                          \
            return _mm256_sub_epi32(_mm256_setzero_si256(), a);                 \
        }                                                                       \
        [[nodiscard]] static reg abs(reg a) noexcept {                          \
            return ABSOP(a);                                                    \
        }                                                                       \
        [[nodiscard]] static reg min(reg a, reg b) noexcept {                   \
            return MINOP(a, b);                                                 \
        }                                                                       \
        [[nodiscard]] static reg max(reg a, reg b) noexcept {                   \
            return MAXOP(a, b);                                                 \
        }                                                                       \
                                                                                \
        [[nodiscard]] static reg band(reg a, reg b) noexcept {                  \
            return _mm256_and_si256(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg bor(reg a, reg b) noexcept {                   \
            return _mm256_or_si256(a, b);                                       \
        }                                                                       \
        [[nodiscard]] static reg bxor(reg a, reg b) noexcept {                  \
            return _mm256_xor_si256(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg bandnot(reg a, reg b) noexcept {               \
            return _mm256_andnot_si256(a, b);                                   \
        }                                                                       \
        [[nodiscard]] static reg shl(reg a, int n) noexcept {                   \
            return _mm256_sllv_epi32(a, _mm256_set1_epi32(n));                  \
        }                                                                       \
        [[nodiscard]] static reg shr_logical(reg a, int n) noexcept {           \
            return _mm256_srlv_epi32(a, _mm256_set1_epi32(n));                  \
        }                                                                       \
        [[nodiscard]] static reg shr_arith(reg a, int n) noexcept {             \
            return _mm256_srav_epi32(a, _mm256_set1_epi32(n));                  \
        }                                                                       \
                                                                                \
        [[nodiscard]] static mask_reg cmpeq(reg a, reg b) noexcept {            \
            return _mm256_cmpeq_epi32(a, b);                                    \
        }                                                                       \
        [[nodiscard]] static mask_reg cmpne(reg a, reg b) noexcept {            \
            return mask_not(_mm256_cmpeq_epi32(a, b));                          \
        }                                                                       \
        [[nodiscard]] static mask_reg cmpgt(reg a, reg b) noexcept {            \
            return CMPGT(a, b);                                                 \
        }                                                                       \
        [[nodiscard]] static mask_reg cmplt(reg a, reg b) noexcept {            \
            return CMPGT(b, a);                                                 \
        }                                                                       \
        [[nodiscard]] static mask_reg cmple(reg a, reg b) noexcept {            \
            return mask_not(CMPGT(a, b));                                       \
        }                                                                       \
        [[nodiscard]] static mask_reg cmpge(reg a, reg b) noexcept {            \
            return mask_not(CMPGT(b, a));                                       \
        }                                                                       \
                                                                                \
        [[nodiscard]] static reg select(mask_reg m, reg a, reg b) noexcept {    \
            return _mm256_blendv_epi8(b, a, m);                                 \
        }                                                                       \
                                                                                \
        [[nodiscard]] static mask_reg mask_true() noexcept {                    \
            return avx2_detail::ones_epi32();                                   \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_false() noexcept {                   \
            return _mm256_setzero_si256();                                      \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_from_bits(std::uint64_t b) noexcept {\
            return avx2_detail::bits_to_mask_epi32(b);                          \
        }                                                                       \
        [[nodiscard]] static std::uint64_t mask_bits(mask_reg m) noexcept {     \
            return avx2_detail::traits<TYPE>::bits(m);                 \
        }                                                                       \
        [[nodiscard]] static bool mask_test(mask_reg m, std::size_t lane) noexcept {\
            return ((mask_bits(m) >> lane) & 1u) != 0;                          \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_and(mask_reg a, mask_reg b) noexcept {\
            return _mm256_and_si256(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_or(mask_reg a, mask_reg b) noexcept {\
            return _mm256_or_si256(a, b);                                       \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_xor(mask_reg a, mask_reg b) noexcept {\
            return _mm256_xor_si256(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_not(mask_reg a) noexcept {           \
            return _mm256_xor_si256(a, avx2_detail::ones_epi32());              \
        }                                                                       \
    };

namespace avx2_detail {
/// Signed 32-bit compare is the hardware instruction; the unsigned one is the
/// same compare after flipping the sign bit of both operands.
[[nodiscard]] inline __m256i cmpgt_epi32(__m256i a, __m256i b) noexcept {
    return _mm256_cmpgt_epi32(a, b);
}
[[nodiscard]] inline __m256i cmpgt_epu32(__m256i a, __m256i b) noexcept {
    const __m256i s = signbit_epi32();
    return _mm256_cmpgt_epi32(_mm256_xor_si256(a, s), _mm256_xor_si256(b, s));
}
} // namespace avx2_detail

VECTIS_AVX2_I32(std::int32_t,
                _mm256_min_epi32, _mm256_max_epi32,
                avx2_detail::cmpgt_epi32,
                _mm256_abs_epi32)
VECTIS_AVX2_I32(std::uint32_t,
                _mm256_min_epu32, _mm256_max_epu32,
                avx2_detail::cmpgt_epu32,
                avx2_detail::abs_epu32)

#undef VECTIS_AVX2_I32

// ===========================================================================
// 64-bit integers.  A deliberate subset: AVX2 has no 64-bit multiply, no
// 64-bit min/max and no 64-bit ordered compare.  Those operations are simply
// absent here rather than emulated slowly, so a kernel that needs them fails
// to compile instead of quietly running at a quarter speed.
// ===========================================================================
#define VECTIS_AVX2_I64(TYPE, SET1, ABSOP)                                       \
    template <> struct backend<TYPE, avx2_abi> {                                \
        using value_type = TYPE;                                                \
        using reg        = __m256i;                                             \
        using mask_reg   = __m256i;                                             \
        static constexpr std::size_t lanes = 4;                                 \
                                                                                \
        [[nodiscard]] static reg zero() noexcept {                              \
            return _mm256_setzero_si256();                                      \
        }                                                                       \
        [[nodiscard]] static reg set1(TYPE v) noexcept {                        \
            return SET1(v);                                                     \
        }                                                                       \
        [[nodiscard]] static reg load(const TYPE* p) noexcept {                 \
            return avx2_detail::loadu_epi(p);                                   \
        }                                                                       \
        static void store(TYPE* p, reg v) noexcept {                            \
            avx2_detail::storeu_epi(p, v);                                      \
        }                                                                       \
        [[nodiscard]] static reg iota() noexcept {                              \
            return avx2_detail::traits<TYPE>::iota();               \
        }                                                                       \
                                                                                \
        [[nodiscard]] static reg add(reg a, reg b) noexcept {                   \
            return _mm256_add_epi64(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg sub(reg a, reg b) noexcept {                   \
            return _mm256_sub_epi64(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg neg(reg a) noexcept {                          \
            return _mm256_sub_epi64(_mm256_setzero_si256(), a);                 \
        }                                                                       \
        [[nodiscard]] static reg abs(reg a) noexcept {                          \
            return ABSOP(a);                                                    \
        }                                                                       \
        [[nodiscard]] static reg band(reg a, reg b) noexcept {                  \
            return _mm256_and_si256(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg bor(reg a, reg b) noexcept {                   \
            return _mm256_or_si256(a, b);                                       \
        }                                                                       \
        [[nodiscard]] static reg bxor(reg a, reg b) noexcept {                  \
            return _mm256_xor_si256(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static reg bandnot(reg a, reg b) noexcept {               \
            return _mm256_andnot_si256(a, b);                                   \
        }                                                                       \
        [[nodiscard]] static reg shl(reg a, int n) noexcept {                   \
            return _mm256_sllv_epi64(a, _mm256_set1_epi64x(n));                 \
        }                                                                       \
        [[nodiscard]] static reg shr_logical(reg a, int n) noexcept {           \
            return _mm256_srlv_epi64(a, _mm256_set1_epi64x(n));                 \
        }                                                                       \
        [[nodiscard]] static reg shr_arith(reg a, int n) noexcept {             \
            return avx2_detail::sra_epi64(a, n);                                \
        }                                                                       \
                                                                                \
        [[nodiscard]] static mask_reg cmpeq(reg a, reg b) noexcept {            \
            return _mm256_cmpeq_epi64(a, b);                                    \
        }                                                                       \
        [[nodiscard]] static mask_reg cmpne(reg a, reg b) noexcept {            \
            return mask_not(_mm256_cmpeq_epi64(a, b));                          \
        }                                                                       \
                                                                                \
        [[nodiscard]] static reg select(mask_reg m, reg a, reg b) noexcept {    \
            return _mm256_blendv_epi8(b, a, m);                                 \
        }                                                                       \
                                                                                \
        [[nodiscard]] static mask_reg mask_true() noexcept {                    \
            return avx2_detail::ones_epi32();                                   \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_false() noexcept {                   \
            return _mm256_setzero_si256();                                      \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_from_bits(std::uint64_t b) noexcept {\
            return avx2_detail::bits_to_mask_epi64(b);                          \
        }                                                                       \
        /* 64-bit lanes move through movemask_pd: 4 bits, one per lane. */      \
        [[nodiscard]] static std::uint64_t mask_bits(mask_reg m) noexcept {     \
            return static_cast<std::uint64_t>(                                  \
                _mm256_movemask_pd(_mm256_castsi256_pd(m)));                    \
        }                                                                       \
        [[nodiscard]] static bool mask_test(mask_reg m, std::size_t lane) noexcept {\
            return ((mask_bits(m) >> lane) & 1u) != 0;                          \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_and(mask_reg a, mask_reg b) noexcept {\
            return _mm256_and_si256(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_or(mask_reg a, mask_reg b) noexcept {\
            return _mm256_or_si256(a, b);                                       \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_xor(mask_reg a, mask_reg b) noexcept {\
            return _mm256_xor_si256(a, b);                                      \
        }                                                                       \
        [[nodiscard]] static mask_reg mask_not(mask_reg a) noexcept {           \
            return _mm256_xor_si256(a, avx2_detail::ones_epi32());              \
        }                                                                       \
    };

VECTIS_AVX2_I64(std::int64_t,
                avx2_detail::set1_epi64,
                avx2_detail::abs_epi64)
VECTIS_AVX2_I64(std::uint64_t,
                avx2_detail::set1_epu64,
                avx2_detail::abs_epu64)

#undef VECTIS_AVX2_I64

} // namespace vectis

#endif // VECTIS_USE_AVX2
