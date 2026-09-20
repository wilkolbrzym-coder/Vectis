// ===========================================================================
// Vectis - SoA conversion and vectorised kernels over it
// ===========================================================================
#include "vectis_test.hpp"

#include <vectis/core/cpu.hpp>
#include <vectis/simd/vec.hpp>
#include <vectis/soa/soa.hpp>

#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

using namespace vectis;

namespace {

// A struct of the kind a particle system actually has.
struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float life;
};

// The layout, named the way the container needs it: one pointer-to-member per
// field.  No macro, nothing to keep in sync with the struct definition.
using F_x    = field_of<&Particle::x>;
using F_y    = field_of<&Particle::y>;
using F_z    = field_of<&Particle::z>;
using F_vx   = field_of<&Particle::vx>;
using F_vy   = field_of<&Particle::vy>;
using F_vz   = field_of<&Particle::vz>;
using F_life = field_of<&Particle::life>;

using ParticleLayout = mem_layout<Particle, &Particle::x, &Particle::y,
                              &Particle::z, &Particle::vx, &Particle::vy,
                              &Particle::vz, &Particle::life>;

std::vector<Particle> make_particles(std::size_t n) {
    std::vector<Particle> p(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float f = static_cast<float>(i);
        p[i].x = f * 0.37f - 5.0f;
        p[i].y = f * -0.11f + 2.0f;
        p[i].z = f * 0.05f;
        p[i].vx = f * 0.01f + 0.3f;
        p[i].vy = f * -0.02f;
        p[i].vz = f * 0.004f - 0.1f;
        p[i].life = 1.0f - f * 0.001f;
    }
    return p;
}

/// The scalar kernel: one struct at a time, the way it is written before
/// anyone reaches for SIMD.  This is the oracle.
void integrate_aos(Particle* p, std::size_t n, float dt, float damping) {
    for (std::size_t i = 0; i < n; ++i) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
        p[i].vx *= damping;
        p[i].vy *= damping;
        p[i].vz *= damping;
        p[i].life -= dt;
    }
}

/// The same kernel over SoA storage, N lanes at a time.
///
/// Note the tail: for_each_block stops at the last whole block and hands the
/// offset back, so the caller writes the scalar epilogue.  That is deliberate -
/// only the caller knows what the kernel means.
void integrate_soa(soa_array<ParticleLayout>& a, float dt, float damping) {
    constexpr std::size_t N = 16;
    using V = f32<N>;
    const V vdt = V::set1(dt);
    const V vdamp = V::set1(damping);

    const std::size_t done = a.for_each_block<N>([&](std::size_t i) {
        a.store<F_x, N>(i, a.load<F_x, N>(i) + a.load<F_vx, N>(i) * vdt);
        a.store<F_y, N>(i, a.load<F_y, N>(i) + a.load<F_vy, N>(i) * vdt);
        a.store<F_z, N>(i, a.load<F_z, N>(i) + a.load<F_vz, N>(i) * vdt);
        a.store<F_vx, N>(i, a.load<F_vx, N>(i) * vdamp);
        a.store<F_vy, N>(i, a.load<F_vy, N>(i) * vdamp);
        a.store<F_vz, N>(i, a.load<F_vz, N>(i) * vdamp);
        a.store<F_life, N>(i, a.load<F_life, N>(i) - vdt);
    });

    // Scalar epilogue for the ragged end.
    auto* px = a.data<F_x>(); auto* py = a.data<F_y>(); auto* pz = a.data<F_z>();
    auto* vx = a.data<F_vx>(); auto* vy = a.data<F_vy>(); auto* vz = a.data<F_vz>();
    auto* lf = a.data<F_life>();
    for (std::size_t i = done; i < a.size(); ++i) {
        px[i] += vx[i] * dt; py[i] += vy[i] * dt; pz[i] += vz[i] * dt;
        vx[i] *= damping;    vy[i] *= damping;    vz[i] *= damping;
        lf[i] -= dt;
    }
}

} // namespace

// ===========================================================================
// The layout machinery itself
// ===========================================================================

VECTIS_TEST(soa_layout_metadata) {
    CHECK_EQ(ParticleLayout::count(), 7u);
    CHECK_EQ(ParticleLayout::index_of<F_x>(), 0u);
    CHECK_EQ(ParticleLayout::index_of<F_z>(), 2u);
    CHECK_EQ(ParticleLayout::index_of<F_life>(), 6u);
    // A field not in the layout reports count(), not 0: an index of 0 would be
    // a silent aliasing bug in every kernel that used it.
    struct Other { float q; };
    CHECK_EQ(ParticleLayout::index_of<field_of<&Other::q>>(),
             ParticleLayout::count());

    // The descriptor interface is what the container depends on, so it is worth
    // asserting that it is complete.
    Particle p{1, 2, 3, 4, 5, 6, 7};
    CHECK_NEAR(F_x::get(p), 1.0f, 0.0);
    CHECK_NEAR(F_life::get(p), 7.0f, 0.0);
    F_z::set(p, 42.0f);
    CHECK_NEAR(F_z::get(p), 42.0f, 0.0);
    static_assert(FieldOf<F_x, Particle>);
    static_assert(std::same_as<F_vx::type, float>);
}

// ===========================================================================
// The conversion round trip
// ===========================================================================

VECTIS_TEST(soa_aos_roundtrip_is_exact) {
    for (std::size_t n : {0u, 1u, 7u, 64u, 1000u}) {
        const auto src = make_particles(n);
        soa_array<ParticleLayout> arr(src.data(), n);
        CHECK_EQ(arr.size(), n);

        // Field storage is contiguous and correctly sized - that is what makes
        // the gather a plain vector load.
        CHECK_EQ(arr.field_bytes<F_x>(), n * sizeof(float));

        std::vector<Particle> back(n);
        arr.to_aos(back.data());
        for (std::size_t i = 0; i < n; ++i) {
            CHECK_ULP(back[i].x, src[i].x, 0);
            CHECK_ULP(back[i].y, src[i].y, 0);
            CHECK_ULP(back[i].z, src[i].z, 0);
            CHECK_ULP(back[i].vx, src[i].vx, 0);
            CHECK_ULP(back[i].vy, src[i].vy, 0);
            CHECK_ULP(back[i].vz, src[i].vz, 0);
            CHECK_ULP(back[i].life, src[i].life, 0);
        }
    }
}

// ===========================================================================
// A real kernel: the SoA result must match the AoS result
// ===========================================================================

VECTIS_TEST(soa_kernel_matches_aos_oracle) {
    // Every length that stresses a different tail residue against N=16.
    for (std::size_t n = 0; n <= 80; ++n) {
        constexpr float dt = 0.125f;
        constexpr float damping = 0.998f;

        auto aos = make_particles(n);
        auto expected = aos;
        integrate_aos(expected.data(), n, dt, damping);

        soa_array<ParticleLayout> soa(aos.data(), n);
        integrate_soa(soa, dt, damping);

        std::vector<Particle> got(n);
        soa.to_aos(got.data());

        for (std::size_t i = 0; i < n; ++i) {
            // The order of operations is the same in both kernels, so these
            // must agree exactly, not approximately.
            CHECK_ULP(got[i].x, expected[i].x, 0);
            CHECK_ULP(got[i].y, expected[i].y, 0);
            CHECK_ULP(got[i].z, expected[i].z, 0);
            CHECK_ULP(got[i].vx, expected[i].vx, 0);
            CHECK_ULP(got[i].vy, expected[i].vy, 0);
            CHECK_ULP(got[i].vz, expected[i].vz, 0);
            CHECK_ULP(got[i].life, expected[i].life, 0);
        }
    }
}

// ===========================================================================
// The bounds are the easy thing to get wrong
// ===========================================================================

VECTIS_TEST(soa_kernel_touches_nothing_past_the_end) {
    // A store that runs past N would show up as a corrupted sentinel, without
    // needing a sanitizer to catch it.
    constexpr std::size_t n = 70;         // 4 blocks of 16, tail of 6
    auto src = make_particles(n + 8);

    soa_array<ParticleLayout> arr(src.data(), n);

    constexpr std::size_t N = 16;
    const std::size_t done = arr.for_each_block<N>([&](std::size_t i) {
        arr.store<F_x, N>(i, arr.load<F_x, N>(i) + f32<N>::set1(1.0f));
    });
    CHECK_EQ(done, 64u);
    CHECK_EQ(n - done, 6u);

    // Every element before the tail moved by exactly one...
    const auto x = arr.data<F_x>();
    for (std::size_t i = 0; i < done; ++i) CHECK_ULP(x[i], src[i].x + 1.0f, 0);
    // ...and nothing at or after it was written at all.
    for (std::size_t i = done; i < n; ++i) CHECK_ULP(x[i], src[i].x, 0);
}

/// Every whole block, and no partial one, on every length.
///
/// The count was previously asserted only for n = 70 - a length with a ragged
/// tail, where `i + N <= size_` and the subtly wrong `i + N < size_` agree.  A
/// loop that skipped the last complete block therefore passed the whole suite,
/// and the kernel test could not see it either, because integrate_soa's scalar
/// epilogue recomputes the block the vector loop dropped.  Lengths that are
/// exact multiples of N are the ones that tell the two apart.
VECTIS_TEST(soa_for_each_block_covers_every_whole_block) {
    constexpr std::size_t N = 16;
    for (const std::size_t n : {0u, 1u, 15u, 16u, 17u, 31u, 32u, 33u, 48u, 64u, 128u}) {
        auto src = make_particles(n);
        soa_array<ParticleLayout> arr(src.data(), n);

        std::size_t blocks = 0;
        std::size_t first = 0;
        std::size_t last = 0;
        const std::size_t done = arr.for_each_block<N>([&](std::size_t i) {
            if (blocks == 0) first = i;
            last = i;
            ++blocks;
        });

        CHECK_EQ(done, (n / N) * N);
        CHECK_EQ(blocks, n / N);
        if (blocks != 0) {
            CHECK_EQ(first, 0u);
            CHECK_EQ(last, ((n / N) - 1) * N);
        }
    }
}

// ===========================================================================
// Value semantics
// ===========================================================================

/// The one invariant this container has is `size_ == every array's size`, and a
/// move is where it used to break.
///
/// With the special members defaulted, a move left every `std::vector` empty
/// but copied `size_` across, so the moved-from object reported the pre-move
/// size through size(), empty() and field_bytes() while `data<F>()` returned
/// nullptr - and `to_aos` below then dereferenced eight null pointers.  The
/// container is left empty and consistent now, as the standard containers are.
VECTIS_TEST(soa_move_leaves_a_consistent_container) {
    auto src = make_particles(8);
    soa_array<ParticleLayout> a(src.data(), 8);

    soa_array<ParticleLayout> b(std::move(a));

    CHECK_EQ(b.size(), 8u);
    CHECK_EQ(a.size(), 0u);
    CHECK(a.empty());
    CHECK_EQ(a.field_bytes<F_x>(), 0u);

    // Reading the moved-from container must be a no-op rather than a fault.
    std::vector<Particle> scratch(8);
    a.to_aos(scratch.data());

    // And the moved-to container owns the values.
    std::vector<Particle> back(8);
    b.to_aos(back.data());
    for (std::size_t i = 0; i < 8; ++i) {
        CHECK_ULP(back[i].x, src[i].x, 0);
        CHECK_ULP(back[i].life, src[i].life, 0);
    }
}

VECTIS_TEST(soa_move_assign_and_swap_keep_size_in_step) {
    auto s1 = make_particles(5);
    auto s2 = make_particles(9);

    soa_array<ParticleLayout> a(s1.data(), 5);
    soa_array<ParticleLayout> b(s2.data(), 9);

    a = std::move(b);
    CHECK_EQ(a.size(), 9u);
    CHECK_EQ(b.size(), 0u);
    CHECK(b.empty());

    soa_array<ParticleLayout> c(s1.data(), 5);
    using std::swap;
    swap(a, c);
    CHECK_EQ(a.size(), 5u);
    CHECK_EQ(c.size(), 9u);

    std::vector<Particle> back(9);
    c.to_aos(back.data());
    for (std::size_t i = 0; i < 9; ++i) CHECK_ULP(back[i].x, s2[i].x, 0);

    std::vector<Particle> back5(5);
    a.to_aos(back5.data());
    for (std::size_t i = 0; i < 5; ++i) CHECK_ULP(back5[i].x, s1[i].x, 0);
}

/// A copy is independent of its source, and resizing one does not touch the
/// other.
VECTIS_TEST(soa_copy_is_independent) {
    auto src = make_particles(6);
    soa_array<ParticleLayout> a(src.data(), 6);
    soa_array<ParticleLayout> b = a;

    CHECK_EQ(b.size(), 6u);
    b.data<F_x>()[0] = 999.0f;
    CHECK_ULP(a.data<F_x>()[0], src[0].x, 0);

    b.resize(3);
    CHECK_EQ(b.size(), 3u);
    CHECK_EQ(a.size(), 6u);
}
