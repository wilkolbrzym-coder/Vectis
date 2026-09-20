// ===========================================================================
// Vectis - the ISA-specific test batteries, behind a portable interface
// ===========================================================================
//
// WHY THIS HEADER EXISTS, because the reason is not obvious and it cost a CI
// run to learn:
//
//   A translation unit compiled with -mavx512f may contain AVX-512
//   instructions in *any* function in it - including a function that only
//   checks whether AVX-512 is available.  A runtime guard inside such a TU is
//   therefore not a guard at all: the check itself can fault before it returns
//   false.  The same TU also runs its static initialisers before main, on a CPU
//   that may not support the instructions they were compiled with.
//
// So the ISA-specific translation units contain NOTHING but these two exported
// functions.  No test registrations, no globals with dynamic initialisation, no
// capability checks.  The portable translation unit owns the decision: it asks
// cpu::has(), and only then calls in here.
//
// The rule generalises to any consumer: put the capability check in code that
// was compiled for the lowest tier you support, and cross into the high-tier
// code through a call.
//
// ===========================================================================
#pragma once

namespace vtest_batteries {

/// Run the full AVX2 battery.  Defined in a TU compiled for AVX2; a no-op
/// elsewhere.  Call only after cpu::has(isa_level::avx2).
void run_avx2();

/// Run the full AVX-512 battery.  Defined in a TU compiled for AVX-512; a no-op
/// elsewhere.  Call only after cpu::has(isa_level::avx512), which requires the
/// full F/BW/DQ/VL quartet.
void run_avx512();

/// Just the out-of-bounds canaries for AVX2.  Separate from the full battery so
/// `ctest -R canary` stays a fast, focused check rather than a duplicate run.
void run_avx2_canaries();

/// Same, for AVX-512.
void run_avx512_canaries();

/// True when this build actually compiled the AVX2 battery in, as opposed to
/// stubbing it out.  Lets a test report "not built" rather than "not supported"
/// - two very different messages for someone reading a CI log.
[[nodiscard]] bool avx2_battery_built() noexcept;

/// Same, for AVX-512.
[[nodiscard]] bool avx512_battery_built() noexcept;

} // namespace vtest_batteries
