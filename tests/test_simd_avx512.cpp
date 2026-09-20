// ===========================================================================
// Vectis - AVX-512 battery, in its own translation unit
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
// has it, regardless of which tier the library itself was built for.  On
// GitHub's Xeon Platinum runners that is real coverage of the widest tier,
// which is exactly the hardware the machine in front of you does not have.
//
// ===========================================================================
#include "simd_battery.hpp"
#include "vectis_test.hpp"

#include <vectis/core/cpu.hpp>
#include <vectis/simd/reduce.hpp>
#include <vectis/simd/vec.hpp>

#include <cstdio>

using namespace vectis;

#if defined(VECTIS_USE_AVX512)

namespace {

void run_avx512_all() {
    vtest_battery::full_battery<avx512_abi>();
    vtest_battery::canaries<avx512_abi>();
    vtest_battery::nan_and_zero_semantics<avx512_abi>();
    vtest_battery::rsqrt_accuracy<avx512_abi>("avx512", 4);
    vtest_battery::rcp_accuracy<avx512_abi>("avx512", 4);
    vtest_battery::padding_never_observed<avx512_abi>();
}

bool host_has_avx512() {
    return cpu::has(isa_level::avx512);
}

} // namespace

VECTIS_TEST(simd_core_avx512) {
    if (!host_has_avx512()) {
        std::printf("      skipped: host has no AVX-512 (compiled and linked, "
                    "never executed here)\n");
        return;
    }
    run_avx512_all();
}

VECTIS_TEST(simd_canary_avx512) {
    if (!host_has_avx512()) {
        std::printf("      skipped: host has no AVX-512\n");
        return;
    }
    vtest_battery::canaries<avx512_abi>();
}

/// Things that are only true on the widest tier, asserted where they can be.
VECTIS_TEST(simd_avx512_specific_properties) {
    if (!host_has_avx512()) {
        std::printf("      skipped: host has no AVX-512\n");
        return;
    }

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
    for (std::size_t i = 0; i < 8; ++i) CHECK_EQ(mn[i], a[i] < b[i] ? a[i] : b[i]);

    const auto lt = (I64::load(a.data()) < I64::load(b.data())).bits();
    std::uint64_t want = 0;
    for (std::size_t i = 0; i < 8; ++i) if (a[i] < b[i]) want |= (1ull << i);
    CHECK_EQ(lt, want);
}

#else

// The build did not enable AVX-512, so there is nothing to instantiate.  Say so
// rather than silently having no coverage.
VECTIS_TEST(simd_core_avx512) {
    std::printf("      skipped: this translation unit was not built with "
                "AVX-512 flags\n");
}

#endif
