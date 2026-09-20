// ===========================================================================
// Vectis - AVX2 battery (compiled for AVX2; see backend_batteries.hpp)
// ===========================================================================
//
// Compiled with `-mavx2 -mfma` and with AVX-512 explicitly disabled, so it is a
// pure AVX2 translation unit regardless of which tier the library was built
// for.  Entered only after cpu::has(isa_level::avx2).
//
// The symmetry with test_simd_avx512.cpp is the point: between the two files
// and the portable one, a single test binary exercises the scalar oracle, the
// 256-bit backend and the 512-bit backend on any host, whatever tier the
// library itself was configured for.  A scalar-tier build still proves the AVX2
// kernel is correct; it just cannot run all of it.
//
// What is asserted here and nowhere else: the AVX2 64-bit integer gaps.  They
// are real limitations of the instruction set, and the capability concepts that
// expose them are load-bearing - a kernel constrained on BackendHasMul<int64,
// avx2> must not exist.  Asserting their absence keeps someone from "fixing" it
// later by adding a slow emulation that quietly halves throughput.
//
// ===========================================================================
#include "backend_batteries.hpp"
#include "simd_battery.hpp"

#include <vectis/core/cpu.hpp>
#include <vectis/simd/reduce.hpp>
#include <vectis/simd/vec.hpp>

#include <cstdint>

using namespace vectis;

// ---------------------------------------------------------------------------
// Compile-time assertions.  Namespace scope, so they are checked whenever this
// TU is compiled and generate no code.
// ---------------------------------------------------------------------------
#if defined(VECTIS_USE_AVX2)

// The multi-register fold, at this specific width.
static_assert(basic_vec<float, 5, avx2_abi>::num_regs == 1);
static_assert(basic_vec<float, 5, avx2_abi>::full_regs == 0);
static_assert(basic_vec<float, 5, avx2_abi>::tail_lanes == 5);
static_assert(basic_vec<float, 8, avx2_abi>::num_regs == 1);
static_assert(basic_vec<float, 16, avx2_abi>::num_regs == 2);
static_assert(basic_vec<float, 16, avx2_abi>::full_regs == 2);
static_assert(basic_vec<float, 16, avx2_abi>::tail_lanes == 0);
static_assert(basic_vec<float, 17, avx2_abi>::num_regs == 3);
static_assert(basic_vec<float, 17, avx2_abi>::tail_lanes == 1);
static_assert(basic_vec<double, 4, avx2_abi>::num_regs == 1);
static_assert(basic_vec<double, 8, avx2_abi>::num_regs == 2);
static_assert(basic_vec<std::int32_t, 8, avx2_abi>::num_regs == 1);
static_assert(basic_vec<std::int64_t, 4, avx2_abi>::num_regs == 1);

// What AVX2 can do...
static_assert(BackendHasMul<float, avx2_abi>);
static_assert(BackendHasFma<float, avx2_abi>);
static_assert(BackendHasSqrt<double, avx2_abi>);
static_assert(BackendHasOrdering<float, avx2_abi>);
static_assert(BackendHasShifts<std::int32_t, avx2_abi>);
static_assert(BackendHasReciprocal<float, avx2_abi>);

// ...and what it cannot.  These three absences are why the capability concepts
// exist at all, and asserting them keeps someone from "fixing" the gap later
// with a slow emulation that quietly halves throughput.
static_assert(!BackendHasMul<std::int64_t, avx2_abi>);
static_assert(!BackendHasMinMax<std::int64_t, avx2_abi>);
static_assert(!BackendHasOrdering<std::int64_t, avx2_abi>);
// Shifts and equality are still there, which makes the i64 backend a subset
// rather than a hole.
static_assert(BackendHasShifts<std::int64_t, avx2_abi>);

namespace vtest_batteries {

void run_avx2() {
    vtest_battery::full_battery<avx2_abi>();
    vtest_battery::canaries<avx2_abi>();
    vtest_battery::nan_and_zero_semantics<avx2_abi>();
    vtest_battery::rsqrt_accuracy<avx2_abi>("avx2", 4);
    vtest_battery::rcp_accuracy<avx2_abi>("avx2", 4);
    vtest_battery::padding_never_observed<avx2_abi>();

    // abs() on an unsigned lane is the identity, and must match the oracle.
    // This was a real bug: the signed vpabsd was being used for uint32 lanes,
    // so (uint32)-4 came back as 4 instead of 2^32-4.
    using U64 = basic_vec<std::uint64_t, 4, avx2_abi>;
    const std::array<std::uint64_t, 4> in{0u, 1u, 0x8000'0000'0000'0000ull,
                                          0xFFFF'FFFF'FFFF'FFFFull};
    const auto got = U64::load(in.data()).abs().to_array();
    for (std::size_t i = 0; i < 4; ++i) CHECK_EQ(got[i], in[i]);

    using I64 = basic_vec<std::int64_t, 4, avx2_abi>;
    const std::array<std::int64_t, 4> si{0, -1, -7, 9'000'000'000ll};
    const auto sabs = I64::load(si.data()).abs().to_array();
    for (std::size_t i = 0; i < 4; ++i) {
        CHECK_EQ(sabs[i], si[i] < 0 ? -si[i] : si[i]);
    }
}

void run_avx2_canaries() { vtest_battery::canaries<avx2_abi>(); }

bool avx2_battery_built() noexcept { return true; }

} // namespace vtest_batteries

#else  // no AVX2 in this TU

namespace vtest_batteries {
void run_avx2() {}
void run_avx2_canaries() {}
bool avx2_battery_built() noexcept { return false; }
} // namespace vtest_batteries

#endif
