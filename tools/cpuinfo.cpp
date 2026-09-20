// ===========================================================================
// vectis-cpuinfo - what this machine can do, and what this binary will use
// ===========================================================================
//
// Prints the two halves of the dispatch question side by side: the tier the
// translation unit was compiled for, and the tier the CPU can actually run.
// On a machine that cannot execute the compiled tier, that difference is the
// whole story, and it is worth being able to see it in one command.
//
// ===========================================================================
#include <vectis/core/config.hpp>
#include <vectis/core/cpu.hpp>
#include <vectis/simd/vec.hpp>

#include <cstdio>

namespace {

void row(const char* label, const char* value) {
    std::printf("  %-22s %s\n", label, value);
}

void row_int(const char* label, long long value) {
    std::printf("  %-22s %lld\n", label, value);
}

} // namespace

int main() {
    std::printf("\nvectis-cpuinfo %s\n", VECTIS_VERSION_STRING);
    std::printf("======================\n\n");

    std::printf("host\n");
    {
        const auto b = vectis::cpu::brand();
        std::printf("  %-22s %.*s\n", "cpu", static_cast<int>(b.size()), b.data());
    }
    row("architecture", VECTIS_ARCH_NAME);
    // The compiler's version, not the library's.  VECTIS_VERSION_STRING is the
    // version of Vectis and already appears on the banner line above; this row
    // is about the toolchain, and printing the wrong one here would mislead the
    // CI log reader this tool exists for.
    row("compiler", VECTIS_COMPILER_NAME " " VECTIS_COMPILER_VERSION_STRING);

    std::printf("\nruntime features\n");
    {
        const auto d = vectis::cpu::describe();
        std::printf("  %-22s %.*s\n", "detected",
                    static_cast<int>(d.size()), d.data());
    }
    row("best runtime tier", vectis::to_string(vectis::cpu::get().best));
    row_int("isa level (runtime)",
            static_cast<long long>(vectis::cpu::get().best));

    std::printf("\ncompile-time selection\n");
    row("compiled tier", vectis::to_string(vectis::compiled_isa));
    row_int("VECTIS_ISA_LEVEL", VECTIS_ISA_LEVEL);
    row_int("VECTIS_CEILING_ISA", VECTIS_CEILING_ISA);
    row_int("VECTIS_MAX_ISA_LEVEL", VECTIS_MAX_ISA_LEVEL);
#ifdef VECTIS_USE_AVX512
    row("VECTIS_USE_AVX512", "yes");
#else
    row("VECTIS_USE_AVX512", "no");
#endif
#ifdef VECTIS_USE_AVX2
    row("VECTIS_USE_AVX2", "yes");
#else
    row("VECTIS_USE_AVX2", "no");
#endif

    std::printf("\nnative vector widths\n");
    row_int("native_f32 lanes", static_cast<long long>(vectis::native_f32::lanes));
    row_int("native_f32 registers",
            static_cast<long long>(vectis::native_f32::num_regs));
    row_int("native_f64 lanes", static_cast<long long>(vectis::native_f64::lanes));
    row_int("native_i32 lanes", static_cast<long long>(vectis::native_i32::lanes));
    row_int("native_i64 lanes", static_cast<long long>(vectis::native_i64::lanes));
    {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%zu bytes", sizeof(vectis::native_f32));
        row("sizeof(native_f32)", buf);
    }

    // The one diagnostic worth shouting about: a binary built for a tier this
    // CPU cannot execute will die with SIGILL.  Better to say so here.
    std::printf("\nverdict\n");
    if (!vectis::cpu::has(vectis::compiled_isa)) {
        std::printf("  WARNING: this binary was compiled for %s but the CPU only\n"
                    "           supports %s.  Executing vector code will raise\n"
                    "           SIGILL.  This is expected for a cross-tier CI build\n"
                    "           and is why the test runner skips those tiers.\n",
                    vectis::to_string(vectis::compiled_isa),
                    vectis::to_string(vectis::cpu::get().best));
        return 1;
    }
    std::printf("  ok: binary tier %s is executable on this CPU\n",
                vectis::to_string(vectis::compiled_isa));
    return 0;
}
