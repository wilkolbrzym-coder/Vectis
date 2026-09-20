// ===========================================================================
// Vectis - the assembly/C++ boundary
// ===========================================================================
//
// Three kernels exist in two spellings each: one hand-written in
// src/asm/avx2_kernels.S, one written with intrinsics.  They compute the same
// thing by construction, which is what makes the benchmark a fair comparison
// and what the parity test asserts bit for bit.
//
// The intrinsics twins are deliberately written to the *same* shape as the
// assembly - four accumulators, 32 floats per iteration, FMA taking the second
// operand from memory - so the measurement isolates the one variable that
// matters: who chose the instructions and the register allocation.
//
// Calling convention note: every function here requires AVX2 at runtime.  The
// asm ones will execute AVX2 instructions unconditionally; the intrinsics ones
// are emitted with a target attribute, so they are safe to *call* from anywhere
// but must still be *reached* behind cpu::has(isa_level::avx2).
//
// ===========================================================================
#pragma once

#include "../core/config.hpp"
#include "../core/cpu.hpp"

#include <cmath>
#include <cstddef>

#if defined(VECTIS_ARCH_X86)

extern "C" {
/// sum(a[i] * b[i]) for i < n.  Raw pointer + length, no vector type crossing
/// the boundary: the ABI of a __m256 argument differs between compilers, and
/// there is no reason to take on that problem.
float vectis_asm_dot_f32(const float* a, const float* b, std::size_t n);

/// y[i] += alpha * x[i] for i < n.
void vectis_asm_saxpy_f32(float* y, const float* x, float alpha, std::size_t n);

/// sum(a[i]) for i < n.
float vectis_asm_sum_f32(const float* a, std::size_t n);
}

namespace vectis::kernels {

/// Intrinsics twin of vectis_asm_dot_f32.
VECTIS_TARGET_AVX2 inline float dot_f32_intrinsics(const float* VECTIS_RESTRICT a,
                                                   const float* VECTIS_RESTRICT b,
                                                   std::size_t n) noexcept {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    std::size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m256 a0 = _mm256_loadu_ps(a + i + 0);
        const __m256 a1 = _mm256_loadu_ps(a + i + 8);
        const __m256 a2 = _mm256_loadu_ps(a + i + 16);
        const __m256 a3 = _mm256_loadu_ps(a + i + 24);
        acc0 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b + i + 0), acc0);
        acc1 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b + i + 8), acc1);
        acc2 = _mm256_fmadd_ps(a2, _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(a3, _mm256_loadu_ps(b + i + 24), acc3);
    }

    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0xB1));

    float total = _mm_cvtss_f32(s);
    // std::fma, not `total += a[i]*b[i]`: the assembly tail ends in
    // vfmadd231ss, which fuses the multiply and the add into one rounding.
    // Letting the compiler decide whether to contract would make the two
    // implementations differ by 1-2 ulp for every length with a scalar tail -
    // a real, reproducible discrepancy, and exactly the kind of thing that
    // makes "the asm is equivalent" an untrustworthy claim.
    for (; i < n; ++i) total = std::fma(a[i], b[i], total);
    return total;
}

/// Intrinsics twin of vectis_asm_saxpy_f32.
VECTIS_TARGET_AVX2 inline void saxpy_f32_intrinsics(float* VECTIS_RESTRICT y,
                                                    const float* VECTIS_RESTRICT x,
                                                    float alpha,
                                                    std::size_t n) noexcept {
    const __m256 va = _mm256_set1_ps(alpha);
    std::size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        _mm256_storeu_ps(y + i + 0,
            _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i + 0), _mm256_loadu_ps(y + i + 0)));
        _mm256_storeu_ps(y + i + 8,
            _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i + 8), _mm256_loadu_ps(y + i + 8)));
        _mm256_storeu_ps(y + i + 16,
            _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i + 16), _mm256_loadu_ps(y + i + 16)));
        _mm256_storeu_ps(y + i + 24,
            _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i + 24), _mm256_loadu_ps(y + i + 24)));
    }
    // Fused, to match the vfmadd231ss in the assembly tail.
    for (; i < n; ++i) y[i] = std::fma(alpha, x[i], y[i]);
}

/// Intrinsics twin of vectis_asm_sum_f32.
VECTIS_TARGET_AVX2 inline float sum_f32_intrinsics(const float* VECTIS_RESTRICT a,
                                                   std::size_t n) noexcept {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    std::size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(a + i + 0));
        acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(a + i + 8));
        acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(a + i + 16));
        acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(a + i + 24));
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(a + i + 32));
        acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(a + i + 40));
        acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(a + i + 48));
        acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(a + i + 56));
    }

    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0xB1));

    float total = _mm_cvtss_f32(s);
    for (; i < n; ++i) total += a[i];
    return total;
}

/// The plain C++ spelling, for the "what does the compiler do on its own"
/// baseline.  Written the way a competent programmer would write it before
/// reaching for anything: one accumulator, no unrolling hints.
/// The naive scalar baseline: one accumulator, additions in source order.
///
/// This is what a competent programmer writes first, and it is *slow* - not
/// because it is scalar, but because a single accumulator makes the loop a
/// serial chain of dependent adds, and without -ffast-math the compiler is not
/// allowed to break that chain.  It vectorises the multiplies and then adds the
/// lanes back one at a time, which is the worst of both worlds.
inline float dot_f32_scalar(const float* a, const float* b, std::size_t n) noexcept {
    float total = 0.0f;
    for (std::size_t i = 0; i < n; ++i) total += a[i] * b[i];
    return total;
}

/// The fair scalar baseline: four independent accumulators.
///
/// Still plain C++, still no -ffast-math, still IEEE-exact per accumulation
/// order - the programmer simply chose an order with parallelism in it, which
/// is a right the language grants without any flags.  Comparing SIMD against
/// the naive loop above flatters SIMD enormously; comparing against this one is
/// the only way to see how much of the win is *width* and how much is merely
/// *permission to reassociate*.
inline float dot_f32_scalar_4acc(const float* a, const float* b, std::size_t n) noexcept {
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        t0 += a[i + 0] * b[i + 0];
        t1 += a[i + 1] * b[i + 1];
        t2 += a[i + 2] * b[i + 2];
        t3 += a[i + 3] * b[i + 3];
    }
    for (; i < n; ++i) t0 += a[i] * b[i];
    return (t0 + t1) + (t2 + t3);
}

/// The scalar control for "does a wider engine help at all": four accumulators
/// and a 4-wide inner step, which is exactly what the SIMD version does with
/// four 8-lane registers - just without the lanes.
inline float sum_f32_scalar_4acc(const float* a, std::size_t n) noexcept {
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        t0 += a[i + 0];
        t1 += a[i + 1];
        t2 += a[i + 2];
        t3 += a[i + 3];
    }
    for (; i < n; ++i) t0 += a[i];
    return (t0 + t1) + (t2 + t3);
}

/// The scalar reference, explicitly fused.
///
/// `y[i] += alpha * x[i]` would be the obvious spelling, but whether the
/// compiler contracts that into one rounding is implementation-defined and
/// depends on the optimisation level.  That ambiguity made the parity test fail
/// at -O0 and pass at -O2, for reasons that had nothing to do with either
/// kernel - and when y cancels alpha*x the intermediate rounding error is
/// amplified, so no fixed ulp budget is stable.  Naming std::fma removes the
/// question, and makes the comparison exact: a correctly rounded fma is a
/// correctly rounded fma on every tier.
inline void saxpy_f32_scalar(float* y, const float* x, float alpha,
                             std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) y[i] = std::fma(alpha, x[i], y[i]);
}

inline float sum_f32_scalar(const float* a, std::size_t n) noexcept {
    float total = 0.0f;
    for (std::size_t i = 0; i < n; ++i) total += a[i];
    return total;
}

/// Is the assembly layer linked into this binary?
[[nodiscard]] inline bool asm_kernels_available() noexcept {
#if defined(VECTIS_HAS_ASM_KERNELS)
    return true;
#else
    return false;
#endif
}

/// Whether these kernels may be executed at all on this host.
[[nodiscard]] inline bool runnable_here() noexcept {
    return cpu::has(isa_level::avx2);
}

} // namespace vectis::kernels

#endif // VECTIS_ARCH_X86
