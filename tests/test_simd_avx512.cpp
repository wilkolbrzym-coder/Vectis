// ===========================================================================
// Vectis - AVX-512 battery (compiled for AVX-512; see backend_batteries.hpp)
// ===========================================================================
//
// This file is compiled with `-mavx512f -mavx512bw -mavx512dq -mavx512vl`,
// which means every function in it may contain ZMM instructions.  It is
// therefore only ever *entered* after cpu::has(isa_level::avx512) says the host
// can run them.  On a host without AVX-512 the tests skip, loudly.
//
// Why a separate TU rather than a target attribute: GCC declares some AVX-512
// intrinsics with a different, reduced signature when AVX512F is off, so
// `_mm512_cmplt_ps_mask` is a hard compile error there even inside a function
// nobody calls.  The backend cannot sit unexecuted in an AVX2 binary; it needs
// a translation unit that was compiled for it.
//
// What this buys: one test binary that fully exercises AVX-512 on any host that
// has it, regardless of which tier the library itself was built for.  On the CI
// runners that report AVX-512 - AMD EPYC 7763, Zen 3 - that is real coverage of
// the widest tier, which is exactly the hardware the machine in front of you
// does not have.  The pool is mixed, so it is coverage per run, not per commit.
//
// ===========================================================================
#include "backend_batteries.hpp"
#include "simd_battery.hpp"

#include <vectis/core/cpu.hpp>
#include <vectis/simd/reduce.hpp>
#include <vectis/simd/vec.hpp>

#include <cstdint>

using namespace vectis;

#if defined(VECTIS_USE_AVX512)

// Avalanche of shape facts that only exist at this width.
static_assert(basic_vec<float, 8, avx512_abi>::num_regs == 1);
static_assert(basic_vec<float, 8, avx512_abi>::tail_lanes == 8);
static_assert(basic_vec<float, 16, avx512_abi>::num_regs == 1);
static_assert(basic_vec<float, 16, avx512_abi>::tail_lanes == 0);
static_assert(basic_vec<float, 32, avx512_abi>::num_regs == 2);
static_assert(basic_vec<double, 8, avx512_abi>::num_regs == 1);
static_assert(basic_vec<std::int32_t, 16, avx512_abi>::num_regs == 1);
static_assert(basic_vec<std::int64_t, 8, avx512_abi>::num_regs == 1);

// The 64-bit integer operations AVX2 lacks, which AVX-512 supplies in full.
static_assert(BackendHasMul<std::int64_t, avx512_abi>);
static_assert(BackendHasMinMax<std::int64_t, avx512_abi>);
static_assert(BackendHasOrdering<std::int64_t, avx512_abi>);
static_assert(BackendHasReciprocal<double, avx512_abi>);

namespace vtest_batteries {

void run_avx512() {
    vtest_battery::full_battery<avx512_abi>();
    vtest_battery::canaries<avx512_abi>();
    vtest_battery::nan_and_zero_semantics<avx512_abi>();
    vtest_battery::rsqrt_accuracy<avx512_abi>("avx512", 4);
    vtest_battery::rcp_accuracy<avx512_abi>("avx512", 4);
    vtest_battery::padding_never_observed<avx512_abi>();

    // The same regression batteries every tier runs.  Called here even though
    // this TU cannot execute on most hosts: the day it does run, on a runner
    // with AVX-512, the AVX-512 mask and reduction paths get them too.
    vtest_battery::mask_factories<avx512_abi>();
    vtest_battery::mask_equality_is_observable_equality<avx512_abi>();
    vtest_battery::reduce_agrees_with_oracle<avx512_abi>();
    vtest_battery::integer_wrap_semantics<avx512_abi>();
    vtest_battery::reciprocal_endpoints<avx512_abi>();
    vtest_battery::per_lane_construction<avx512_abi>();

    // A mask on AVX-512 is a general-purpose integer, so bits() is the identity
    // rather than a movemask.  The observable consequence is that a 16-lane
    // mask carries 16 bits with no truncation anywhere.
    using M16 = basic_vec<float, 16, avx512_abi>::mask_type;
    CHECK_EQ(M16::full().bits(), 0xFFFFull);
    CHECK_EQ(M16::lane_mask(), 0xFFFFull);
    CHECK_EQ(M16::from_bits(0xABCD).bits(), 0xABCDull);
    CHECK_EQ(M16::from_bits(0xABCD).count(), 10u);

    using M8 = basic_vec<double, 8, avx512_abi>::mask_type;
    CHECK_EQ(M8::full().bits(), 0xFFull);
    CHECK_EQ(M8::from_bits(0xA5).count(), 4u);

    // The 64-bit integer operations AVX2 does not have, exercised for real
    // rather than only asserted as available.
    using I64 = basic_vec<std::int64_t, 8, avx512_abi>;
    const std::array<std::int64_t, 8> a{1, -2, 3, -4, 5, -6, 7, -8};
    const std::array<std::int64_t, 8> b{2, 2, 2, 2, 2, 2, 2, 2};
    const auto prod = (I64::load(a.data()) * I64::load(b.data())).to_array();
    for (std::size_t i = 0; i < 8; ++i) CHECK_EQ(prod[i], a[i] * 2);

    const auto mn = I64::load(a.data()).min(I64::load(b.data())).to_array();
    for (std::size_t i = 0; i < 8; ++i) {
        CHECK_EQ(mn[i], a[i] < b[i] ? a[i] : b[i]);
    }

    const auto lt = (I64::load(a.data()) < I64::load(b.data())).bits();
    std::uint64_t want = 0;
    for (std::size_t i = 0; i < 8; ++i) {
        if (a[i] < b[i]) want |= (std::uint64_t{1} << i);
    }
    CHECK_EQ(lt, want);
}

void run_avx512_canaries() { vtest_battery::canaries<avx512_abi>(); }

bool avx512_battery_built() noexcept { return true; }

} // namespace vtest_batteries

#else

namespace vtest_batteries {
void run_avx512() {}
void run_avx512_canaries() {}
bool avx512_battery_built() noexcept { return false; }
} // namespace vtest_batteries

#endif
