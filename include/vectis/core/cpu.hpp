// ===========================================================================
// Vectis - compile-time SIMD engine
// core/cpu.hpp - runtime CPU feature detection
// ===========================================================================
//
// The compile-time tier (vectis::compiled_isa) says what this binary *may*
// execute.  This header answers the other half of the question: what the CPU
// under us can actually do right now.  Code that ships AVX-512 kernels inside
// an AVX2 binary must gate the call on cpu::has(isa_level::avx512).
//
// Detection is free after the first call: __builtin_cpu_supports reads a table
// the C runtime already built during startup.
// ===========================================================================
#pragma once

#include "config.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace vectis::cpu {

/// A snapshot of the running CPU's SIMD-relevant capabilities.
struct features {
    // baseline x86-64
    bool sse42   : 1;
    bool popcnt  : 1;
    bool avx     : 1;
    bool avx2    : 1;
    bool fma     : 1;
    bool f16c    : 1;
    bool bmi1    : 1;
    bool bmi2    : 1;
    bool movbe   : 1;
    bool adx     : 1;
    // crypto and bit manipulation
    bool aes     : 1;
    bool pclmul  : 1;
    bool sha     : 1;
    bool gfni    : 1;
    bool vaes    : 1;
    bool vpclmul : 1;
    // AVX-512 family
    bool avx512f         : 1;
    bool avx512cd        : 1;
    bool avx512bw        : 1;
    bool avx512dq        : 1;
    bool avx512vl        : 1;
    bool avx512vbmi      : 1;
    bool avx512vbmi2     : 1;
    bool avx512vnni      : 1;
    bool avx512bitalg    : 1;
    bool avx512vpopcntdq : 1;
    bool avx512ifma      : 1;
    bool avx512bf16      : 1;
    bool avx512fp16      : 1;

    /// Highest tier whose *complete* required feature set is present.
    ///
    /// Partial AVX-512 - F without BW/DQ/VL - deliberately reports avx2: the
    /// library's AVX-512 kernels assume the full quartet, so anything less must
    /// not be routed to them.
    isa_level best = isa_level::scalar;
};

#if defined(VECTIS_ARCH_X86) && \
   (defined(VECTIS_COMPILER_GCC) || defined(VECTIS_COMPILER_CLANG))
#  define VECTIS_HAS_CPU_PROBE 1
#endif

namespace detail {

#if defined(VECTIS_ARCH_X86)
inline void cpuid(std::uint32_t leaf, std::uint32_t sub,
                  std::uint32_t& a, std::uint32_t& b,
                  std::uint32_t& c, std::uint32_t& d) noexcept {
    __asm__ __volatile__("cpuid"
                         : "=a"(a), "=b"(b), "=c"(c), "=d"(d)
                         : "a"(leaf), "c"(sub));
}
#endif

#ifdef VECTIS_HAS_CPU_PROBE
// __builtin_cpu_supports takes a string *literal*, so the name cannot be
// funnelled through a variable.  The macro keeps the literal at the call site.
#  define VECTIS_CPU_SUP(name) (__builtin_cpu_supports(name) != 0)
#endif

inline const features& query() noexcept {
    static const features f = [] {
        features r{};
#ifdef VECTIS_HAS_CPU_PROBE
        // The runtime built this table before any constructor ran; force it to
        // materialise even when we are called from an early initialiser.
        __builtin_cpu_init();

        r.sse42   = VECTIS_CPU_SUP("sse4.2");
        r.popcnt  = VECTIS_CPU_SUP("popcnt");
        r.avx     = VECTIS_CPU_SUP("avx");
        r.avx2    = VECTIS_CPU_SUP("avx2");
        r.fma     = VECTIS_CPU_SUP("fma");
        r.f16c    = VECTIS_CPU_SUP("f16c");
        r.bmi1    = VECTIS_CPU_SUP("bmi");
        r.bmi2    = VECTIS_CPU_SUP("bmi2");
        r.movbe   = VECTIS_CPU_SUP("movbe");
        r.adx     = VECTIS_CPU_SUP("adx");
        r.aes     = VECTIS_CPU_SUP("aes");
        r.pclmul  = VECTIS_CPU_SUP("pclmul");
        r.sha     = VECTIS_CPU_SUP("sha");
        r.gfni    = VECTIS_CPU_SUP("gfni");
        r.vaes    = VECTIS_CPU_SUP("vaes");
        r.vpclmul = VECTIS_CPU_SUP("vpclmulqdq");

        r.avx512f    = VECTIS_CPU_SUP("avx512f");
        r.avx512cd   = VECTIS_CPU_SUP("avx512cd");
        r.avx512bw   = VECTIS_CPU_SUP("avx512bw");
        r.avx512dq   = VECTIS_CPU_SUP("avx512dq");
        r.avx512vl   = VECTIS_CPU_SUP("avx512vl");
        r.avx512vbmi = VECTIS_CPU_SUP("avx512vbmi");
        r.avx512vbmi2     = VECTIS_CPU_SUP("avx512vbmi2");
        r.avx512vnni      = VECTIS_CPU_SUP("avx512vnni");
        r.avx512bitalg    = VECTIS_CPU_SUP("avx512bitalg");
        r.avx512vpopcntdq = VECTIS_CPU_SUP("avx512vpopcntdq");
        r.avx512ifma      = VECTIS_CPU_SUP("avx512ifma");
        r.avx512bf16      = VECTIS_CPU_SUP("avx512bf16");
        r.avx512fp16      = VECTIS_CPU_SUP("avx512fp16");

        if (r.avx512f && r.avx512bw && r.avx512dq && r.avx512vl) {
            r.best = isa_level::avx512;
        } else if (r.avx2 && r.fma) {
            r.best = isa_level::avx2;
        } else if (r.sse42) {
            r.best = isa_level::sse42;
        }
#endif
        return r;
    }();
    return f;
}

} // namespace detail

/// The running CPU's features.  Computed once, on first use.
[[nodiscard]] inline const features& get() noexcept { return detail::query(); }

/// True when the running CPU can execute code compiled for `l`.
[[nodiscard]] inline bool has(isa_level l) noexcept {
    return static_cast<int>(get().best) >= static_cast<int>(l);
}

/// True when this translation unit was compiled for `l` *and* the CPU runs it,
/// i.e. the tier the library is actually executing at.
[[nodiscard]] inline bool active(isa_level l) noexcept {
    return compiled_isa == l && has(l);
}

// --------------------------------------------------------------- brand string

namespace detail {

/// 48-byte CPU brand string from CPUID leaves 0x8000'0002..0x8000'0004.
inline std::string_view brand() noexcept {
    static const std::array<char, 49> buf = [] {
        std::array<char, 49> out{};
#if defined(VECTIS_ARCH_X86)
        std::uint32_t a = 0, b = 0, c = 0, d = 0, max_ext = 0;
        cpuid(0x80000000u, 0, max_ext, b, c, d);
        if (max_ext >= 0x80000004u) {
            for (std::uint32_t i = 0; i < 3; ++i) {
                cpuid(0x80000002u + i, 0, a, b, c, d);
                const std::uint32_t regs[4] = {a, b, c, d};
                for (std::uint32_t j = 0; j < 4; ++j) {
                    const auto off = static_cast<std::size_t>(i) * 16u +
                                     static_cast<std::size_t>(j) * 4u;
                    out[off + 0] = static_cast<char>((regs[j] >>  0) & 0xFFu);
                    out[off + 1] = static_cast<char>((regs[j] >>  8) & 0xFFu);
                    out[off + 2] = static_cast<char>((regs[j] >> 16) & 0xFFu);
                    out[off + 3] = static_cast<char>((regs[j] >> 24) & 0xFFu);
                }
            }
            // Intel pads the brand string with leading spaces; shift them out.
            std::size_t lead = 0;
            while (lead < 48 && out[lead] == ' ') ++lead;
            if (lead != 0) {
                for (std::size_t i = 0; i + lead < 49; ++i) {
                    out[i] = out[i + lead];
                }
                for (std::size_t i = 48 - lead; i < 49; ++i) out[i] = '\0';
            }
        }
#endif
        out[48] = '\0';
        return out;
    }();

    std::size_t n = 0;
    while (n < 48 && buf[n] != '\0') ++n;
    return std::string_view{buf.data(), n};
}

} // namespace detail

[[nodiscard]] inline std::string_view brand() noexcept { return detail::brand(); }

/// Human-readable feature summary, e.g. "avx2 fma bmi2 aes pclmulqdq".
[[nodiscard]] inline std::string_view describe() noexcept {
    // The string itself is the static, not a view into a temporary: `out` below
    // must outlive the returned view.
    static const std::string s = [] {
        const features& f = get();
        std::string out;
        out.reserve(192);
        auto add = [&out](const char* n) {
            if (!out.empty()) out += ' ';
            out += n;
        };
        if (f.sse42)   add("sse4.2");
        if (f.avx)     add("avx");
        if (f.avx2)    add("avx2");
        if (f.fma)     add("fma");
        if (f.f16c)    add("f16c");
        if (f.bmi1)    add("bmi1");
        if (f.bmi2)    add("bmi2");
        if (f.movbe)   add("movbe");
        if (f.adx)     add("adx");
        if (f.aes)     add("aes");
        if (f.pclmul)  add("pclmulqdq");
        if (f.sha)     add("sha-ni");
        if (f.gfni)    add("gfni");
        if (f.vaes)    add("vaes");
        if (f.vpclmul) add("vpclmulqdq");
        if (f.avx512f)    add("avx512f");
        if (f.avx512cd)   add("avx512cd");
        if (f.avx512bw)   add("avx512bw");
        if (f.avx512dq)   add("avx512dq");
        if (f.avx512vl)   add("avx512vl");
        if (f.avx512vbmi) add("avx512vbmi");
        if (f.avx512vbmi2) add("avx512vbmi2");
        if (f.avx512vnni) add("avx512vnni");
        if (f.avx512bitalg) add("avx512bitalg");
        if (f.avx512vpopcntdq) add("avx512vpopcntdq");
        if (f.avx512ifma) add("avx512ifma");
        if (f.avx512bf16) add("avx512bf16");
        if (f.avx512fp16) add("avx512fp16");
        return out;
    }();
    return s;
}

} // namespace vectis::cpu
