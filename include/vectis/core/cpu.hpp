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

namespace detail {

#if defined(VECTIS_ARCH_X86)

/// Raw CPUID.  `sub` is the sub-leaf, which only leaf 7 and the extended
/// topology leaves care about.
inline void cpuid(std::uint32_t leaf, std::uint32_t sub,
                                std::uint32_t& a, std::uint32_t& b,
                                std::uint32_t& c, std::uint32_t& d) noexcept {
    __asm__ __volatile__("cpuid"
                         : "=a"(a), "=b"(b), "=c"(c), "=d"(d)
                         : "a"(leaf), "c"(sub));
}

/// Read an extended control register.  Only legal when CPUID.1:ECX.OSXSAVE is
/// set; on a CPU without XSAVE this instruction raises #UD.
[[nodiscard]] inline std::uint64_t xgetbv0() noexcept {
    std::uint32_t lo = 0, hi = 0;
    __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (static_cast<std::uint64_t>(hi) << 32) | lo;
}

[[nodiscard]] inline bool bit(std::uint32_t word, int n) noexcept {
    return ((word >> n) & 1u) != 0;
}

#endif // VECTIS_ARCH_X86

/// Read the CPU's feature set.
///
/// Deliberately built on CPUID directly rather than __builtin_cpu_supports.
/// Two reasons, both found the hard way by CI:
///
///   1. The set of feature *strings* a compiler accepts is not standardised.
///      Clang rejects "movbe", "adx", "sha" and "vaes", which GCC accepts, so
///      the builtin makes the header uncompilable on half the toolchains we
///      claim to support.  CPUID bit positions are architecture, not compiler
///      opinion, and they do not drift.
///   2. The builtin hides the OS-support check.  AVX and AVX-512 need XCR0 to
///      say the kernel saves the wider register state: without that, the first
///      AVX-512 instruction faults or - worse - silently corrupts its
///      neighbour's registers across a context switch.  A SIMD library should
///      be able to point at the check it relies on.
[[nodiscard]] inline features detect() noexcept {
    features r{};

#if defined(VECTIS_ARCH_X86)
    std::uint32_t a = 0, b = 0, c = 0, d = 0;

    cpuid(0, 0, a, b, c, d);
    const std::uint32_t max_leaf = a;
    if (max_leaf < 1) return r;

    // ---- leaf 1: the 1990s-and-2000s set, plus the AVX OS-support bits
    cpuid(1, 0, a, b, c, d);
    r.sse42  = bit(c, 20);
    r.popcnt = bit(c, 23);
    r.movbe  = bit(c, 22);
    r.aes    = bit(c, 25);
    r.pclmul = bit(c, 1);
    r.f16c   = bit(c, 29);
    r.fma    = bit(c, 12);

    const bool has_xsave   = bit(c, 26);
    const bool has_osxsave = bit(c, 27);
    const bool avx_in_hw   = bit(c, 28);

    // xgetbv is only legal once the OS has enabled XSAVE.
    const std::uint64_t xcr0 = (has_xsave && has_osxsave) ? xgetbv0() : 0;
    // bits 1,2  = XMM and YMM state saved
    const bool ymm_ok = (xcr0 & 0x6ull) == 0x6ull;
    // bits 5,6,7 = opmask, ZMM_Hi256, Hi16_ZMM - all three, or AVX-512 is not
    // safe to use even when the CPU reports the instructions.
    const bool zmm_ok = (xcr0 & 0xE6ull) == 0xE6ull;

    r.avx = avx_in_hw && ymm_ok;

    if (max_leaf < 7) {
        if (r.avx2 && r.fma) r.best = isa_level::avx2;
        else if (r.sse42)    r.best = isa_level::sse42;
        return r;
    }

    // ---- leaf 7 sub-leaf 0: BMI, AVX2, and the AVX-512 family
    cpuid(7, 0, a, b, c, d);
    const std::uint32_t max_sub = a;

    r.bmi1 = bit(b, 3);
    r.bmi2 = bit(b, 8);
    r.adx  = bit(b, 19);
    r.sha  = bit(b, 29);
    r.avx2 = bit(b, 5) && r.avx;

    r.avx512f    = bit(b, 16) && zmm_ok;
    r.avx512dq   = bit(b, 17);
    r.avx512ifma = bit(b, 21);
    r.avx512cd   = bit(b, 28);
    r.avx512bw   = bit(b, 30);
    r.avx512vl   = bit(b, 31);

    r.avx512vbmi      = bit(c, 1);
    r.avx512vbmi2     = bit(c, 6);
    r.gfni            = bit(c, 8);
    r.vaes            = bit(c, 9);
    r.vpclmul         = bit(c, 10);
    r.avx512vnni      = bit(c, 11);
    r.avx512bitalg    = bit(c, 12);
    r.avx512vpopcntdq = bit(c, 14);

    r.avx512fp16 = bit(d, 23);

    // ---- leaf 7 sub-leaf 1: AVX512_BF16
    if (max_sub >= 1) {
        cpuid(7, 1, a, b, c, d);
        r.avx512bf16 = bit(a, 5);
    }

    // The 512-bit tier requires the whole quartet, not F alone: byte-granular
    // blends and efficient 32-bit lane compares come from BW and VL, and every
    // kernel in this library assumes they are there.
    if (r.avx512f && r.avx512bw && r.avx512dq && r.avx512vl) {
        r.best = isa_level::avx512;
    } else if (r.avx2 && r.fma) {
        r.best = isa_level::avx2;
    } else if (r.sse42) {
        r.best = isa_level::sse42;
    }
#endif
    return r;
}

inline const features& query() noexcept {
    static const features f = detect();
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
