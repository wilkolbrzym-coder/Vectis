// ===========================================================================
// Vectis - compile-time SIMD engine
// core/config.hpp - compiler, architecture and ISA feature detection
// ===========================================================================
//
// Everything downstream keys off the macros defined here.  The design rule is:
//
//   * what the compiler CAN emit is discovered from the compiler's own
//     predefined macros (-mavx2 defines __AVX2__, and so on),
//   * what we are ALLOWED to emit is capped by VECTIS_MAX_ISA_LEVEL, which the
//     build system sets so the same kernel can be benchmarked at different
//     tiers without editing a single line of source.
//
// ===========================================================================
#pragma once

#define VECTIS_VERSION_MAJOR 0
#define VECTIS_VERSION_MINOR 1
#define VECTIS_VERSION_PATCH 0
#define VECTIS_VERSION_STRING "0.1.0"

// ------------------------------------------------------------------ compiler
//
// Two spellings of the compiler's version: the number, for comparisons, and the
// string, for reports.  A diagnostic that prints the *library* version on a row
// labelled "compiler" is worse than printing nothing, so both are here and the
// tools use the string.
#define VECTIS_DETAIL_STRINGIFY_(x) #x
#define VECTIS_DETAIL_STRINGIFY(x) VECTIS_DETAIL_STRINGIFY_(x)

#if defined(__clang__)
#  define VECTIS_COMPILER_CLANG 1
#  define VECTIS_COMPILER_NAME "clang"
#  define VECTIS_COMPILER_VERSION \
      (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#  define VECTIS_COMPILER_VERSION_STRING \
      VECTIS_DETAIL_STRINGIFY(__clang_major__) "." \
      VECTIS_DETAIL_STRINGIFY(__clang_minor__) "." \
      VECTIS_DETAIL_STRINGIFY(__clang_patchlevel__)
#elif defined(__GNUC__)
#  define VECTIS_COMPILER_GCC 1
#  define VECTIS_COMPILER_NAME "gcc"
#  define VECTIS_COMPILER_VERSION \
      (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#  define VECTIS_COMPILER_VERSION_STRING \
      VECTIS_DETAIL_STRINGIFY(__GNUC__) "." \
      VECTIS_DETAIL_STRINGIFY(__GNUC_MINOR__) "." \
      VECTIS_DETAIL_STRINGIFY(__GNUC_PATCHLEVEL__)
#else
#  error "Vectis: unsupported compiler (GCC or Clang required)"
#endif

// -------------------------------------------------------------- architecture
#if defined(__x86_64__) || defined(_M_X64)
#  define VECTIS_ARCH_X86_64 1
#  define VECTIS_ARCH_NAME "x86_64"
#elif defined(__i386__) || defined(_M_IX86)
#  define VECTIS_ARCH_X86_32 1
#  define VECTIS_ARCH_NAME "x86"
#elif defined(__aarch64__) || defined(_M_ARM64)
#  define VECTIS_ARCH_AARCH64 1
#  define VECTIS_ARCH_NAME "aarch64"
#else
#  define VECTIS_ARCH_UNKNOWN 1
#  define VECTIS_ARCH_NAME "unknown"
#endif

#if defined(VECTIS_ARCH_X86_64) || defined(VECTIS_ARCH_X86_32)
#  define VECTIS_ARCH_X86 1
#endif

// ---------------------------------------------------------------- OS / ABI --
#if defined(__linux__)
#  define VECTIS_OS_LINUX 1
#endif

// --------------------------------------------------- what the compiler can do
// FMA is tracked separately from AVX2 on purpose: AVX2 without FMA exists in
// the wild, and code that assumes the pair silently produces different results
// (no contraction) or fails to compile.
#if defined(__AVX512F__)
#  define VECTIS_CAN_AVX512F 1
#endif
#if defined(__AVX512BW__)
#  define VECTIS_CAN_AVX512BW 1
#endif
#if defined(__AVX512DQ__)
#  define VECTIS_CAN_AVX512DQ 1
#endif
#if defined(__AVX512VL__)
#  define VECTIS_CAN_AVX512VL 1
#endif
#if defined(__AVX2__)
#  define VECTIS_CAN_AVX2 1
#endif
#if defined(__AVX__)
#  define VECTIS_CAN_AVX 1
#endif
#if defined(__FMA__)
#  define VECTIS_CAN_FMA 1
#endif
#if defined(__SSE4_2__)
#  define VECTIS_CAN_SSE42 1
#endif

// A usable AVX-512 baseline needs the whole quartet; F alone cannot express
// byte-granular blends or 32-bit lane compares efficiently.
#if defined(VECTIS_CAN_AVX512F) && defined(VECTIS_CAN_AVX512BW) && \
    defined(VECTIS_CAN_AVX512DQ) && defined(VECTIS_CAN_AVX512VL)
#  define VECTIS_CAN_AVX512 1
#endif

// ------------------------------------------------- what we are allowed to emit
// ISA levels, ordered.  Kept as plain integers so they can be compared with
// < in `#if` and used as non-type template parameters downstream.
#define VECTIS_ISA_SCALAR 0
#define VECTIS_ISA_SSE42  1
#define VECTIS_ISA_AVX2   2
#define VECTIS_ISA_AVX512 3

#ifndef VECTIS_MAX_ISA_LEVEL
#  define VECTIS_MAX_ISA_LEVEL 99
#endif

// The uncapped ceiling implied by the compiler flags.
#if defined(VECTIS_CAN_AVX512)
#  define VECTIS_CEILING_ISA 3
#elif defined(VECTIS_CAN_AVX2) && defined(VECTIS_CAN_FMA)
#  define VECTIS_CEILING_ISA 2
#elif defined(VECTIS_CAN_SSE42)
#  define VECTIS_CEILING_ISA 1
#else
#  define VECTIS_CEILING_ISA 0
#endif

// Effective level = min(ceiling, cap).
#if VECTIS_MAX_ISA_LEVEL < VECTIS_CEILING_ISA
#  define VECTIS_ISA_LEVEL VECTIS_MAX_ISA_LEVEL
#else
#  define VECTIS_ISA_LEVEL VECTIS_CEILING_ISA
#endif

// ------------------------------------------------------ effectful feature tests
// These are what the rest of the library actually branches on.  A feature is
// usable only when it is both emittable and not capped away.
#if VECTIS_ISA_LEVEL >= 3 && defined(VECTIS_CAN_AVX512)
#  define VECTIS_USE_AVX512 1
#endif
#if VECTIS_ISA_LEVEL >= 2 && defined(VECTIS_CAN_AVX2) && defined(VECTIS_CAN_FMA)
#  define VECTIS_USE_AVX2 1
#endif
// There is intentionally no VECTIS_USE_SSE42.  ISA level 1 is an observation
// about the host, not a code generation target: the library has no 128-bit
// backend, so a level-1 build runs the scalar path.  See abi.hpp.

// ------------------------------------------------------------ codegen helpers
#if defined(VECTIS_COMPILER_GCC) || defined(VECTIS_COMPILER_CLANG)
#  define VECTIS_ALWAYS_INLINE inline __attribute__((always_inline))
#  define VECTIS_NOINLINE      __attribute__((noinline))
#  define VECTIS_HOT           __attribute__((hot))
#  define VECTIS_FLATTEN       __attribute__((flatten))
#  define VECTIS_RESTRICT      __restrict__
#  define VECTIS_LIKELY(x)     __builtin_expect(!!(x), 1)
#  define VECTIS_UNLIKELY(x)   __builtin_expect(!!(x), 0)
#  define VECTIS_ASSUME(x)     do { if (!(x)) __builtin_unreachable(); } while (0)
#  define VECTIS_UNREACHABLE() __builtin_unreachable()
#  define VECTIS_PREFETCH(p, rw, locality) \
      __builtin_prefetch((p), (rw), (locality))
#endif

#ifdef VECTIS_ARCH_X86
#  include <x86intrin.h>
#endif

// Compile a function for a specific ISA regardless of the translation unit's
// baseline.  This is how AVX-512 code exists in a binary that only requires
// AVX2 at load time: the function is emitted, and only *called* behind a
// runtime feature check.
#if defined(VECTIS_COMPILER_GCC) || defined(VECTIS_COMPILER_CLANG)
#  define VECTIS_TARGET(...) __attribute__((target(__VA_ARGS__)))
#else
#  define VECTIS_TARGET(...)
#endif

#define VECTIS_TARGET_AVX2   VECTIS_TARGET("avx2,fma")
#define VECTIS_TARGET_AVX512 VECTIS_TARGET("avx512f,avx512bw,avx512dq,avx512vl")

// Multi-versioning: GCC emits one clone per entry and resolves the best one at
// load time via IFUNC.  "default" must come first and means "whatever the
// translation unit was compiled with".
#if defined(VECTIS_COMPILER_GCC) && defined(VECTIS_OS_LINUX)
#  define VECTIS_MULTIVERSION_DEFAULT \
      __attribute__((target_clones("default")))
#  define VECTIS_MULTIVERSION_AVX2 \
      __attribute__((target_clones("default", "avx2,fma")))
#  define VECTIS_MULTIVERSION_FULL \
      __attribute__((target_clones("default", "avx2,fma", \
                                   "avx512f,avx512bw,avx512dq,avx512vl")))
#else
#  define VECTIS_MULTIVERSION_DEFAULT
#  define VECTIS_MULTIVERSION_AVX2
#  define VECTIS_MULTIVERSION_FULL
#endif

// ---------------------------------------------------------------- reflection
// P2996 static reflection.
//
// GCC 16 implements the core behind -freflection, and defines
// __cpp_impl_reflection (202506) when the flag is on.  Without the flag the
// macro is simply absent, so the feature is detectable from a header with no
// build-system cooperation - which is what makes it safe to gate on.
//
// Note that __cpp_lib_meta is NOT defined even when reflection works: the
// library feature-test macro tracks the <meta> header being complete, and it is
// not.  The compiler macro is the one that means "the syntax is available".
#if defined(__cpp_impl_reflection) && __cpp_impl_reflection >= 202506
#  define VECTIS_HAS_REFLECTION 1
#endif

// ------------------------------------------------------------------ sanity ---
#if VECTIS_ISA_LEVEL > 0 && !defined(VECTIS_ARCH_X86)
#  error "Vectis: non-zero ISA level requested on a non-x86 target"
#endif
#if defined(VECTIS_USE_AVX2) && !defined(VECTIS_CAN_FMA)
#  error "Vectis: AVX2 selected without FMA"
#endif

namespace vectis {

/// Ordered ISA tiers.  Mirrors the VECTIS_ISA_* macros so that runtime and
/// compile-time decisions can be written against the same vocabulary.
enum class isa_level : int {
    scalar = 0,
    sse42  = 1,
    avx2   = 2,
    avx512 = 3,
};

/// The tier this translation unit is compiled for.
inline constexpr isa_level compiled_isa =
    static_cast<isa_level>(VECTIS_ISA_LEVEL);

constexpr const char* to_string(isa_level l) noexcept {
    switch (l) {
        case isa_level::scalar: return "scalar";
        case isa_level::sse42:  return "sse4.2";
        case isa_level::avx2:   return "avx2+fma";
        case isa_level::avx512: return "avx512";
    }
    return "?";
}

} // namespace vectis
