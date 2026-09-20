// ===========================================================================
// Vectis - AoS to SoA through real reflection
// ===========================================================================
//
// What this test is defending, stated plainly: the struct below has no
// annotations, no field list, no macro, and no registration of any kind.  If
// reflection stops working - a field is missed, a filtered member leaks back
// in, the order changes - these assertions fail, and they fail in this file
// rather than in a kernel three layers away.
//
// When the compiler has no reflection the whole file compiles to a single skip
// message, so the suite stays green on C++20 toolchains without pretending the
// coverage exists.
// ===========================================================================
#include "vectis_test.hpp"

#include <vectis/core/config.hpp>

#if defined(VECTIS_HAS_REFLECTION)

#include <vectis/simd/vec.hpp>
#include <vectis/soa/reflect.hpp>

#include <cstdio>
#include <string_view>
#include <type_traits>
#include <vector>

using namespace vectis;

namespace {

// Deliberately unannotated.
struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float life;
};

// Members that must be filtered out, mixed in with ones that must survive.
struct Mixed {
    float a;
    int   n;
    char  tag;      // one byte: not a lane type
    double d;
    bool  flag;     // a predicate, not a lane
    long  counter;
};

// The filters that are not about size, two of which were missing and each of
// which failed in a place the caller could not act on: a `const float` member
// passed is_arithmetic_type and became std::vector<const float>, which
// libstdc++ rejects from inside <bits/stl_vector.h>, and a named bit-field
// passed every check and then could not bind a reference in the descriptor
// ("returning reference to temporary") in a header the caller never included.
// Both are silent skips now, and the members around them keep their order.
struct Obstructed {
    unsigned int flags : 3;     // a bit-field is a data member, not a lane
    const float  frozen;        // cv-qualified: std::vector cannot hold it
    float        x;
    volatile int v;
    int          n;
    float*       ptr;           // arithmetic? no - not a lane
    short        s;             // two bytes: this one is a lane
    char         c;             // one byte: not a lane
};

using L      = reflected_layout<Particle>;
using F_x    = L::field_at<0>;
using F_y    = L::field_at<1>;
using F_z    = L::field_at<2>;
using F_vx   = L::field_at<3>;
using F_vy   = L::field_at<4>;
using F_vz   = L::field_at<5>;
using F_life = L::field_at<6>;

constexpr std::size_t N = 64;

std::vector<Particle> make_aos() {
    std::vector<Particle> v(N);
    for (std::size_t i = 0; i < N; ++i) {
        const float f = static_cast<float>(i);
        v[i] = Particle{f, f * 2.0f, f * 3.0f, f * 4.0f,
                        f * 5.0f, f * 6.0f, 1.0f - f * 0.01f};
    }
    return v;
}

} // namespace

// ===========================================================================
// What reflection found
// ===========================================================================

VECTIS_TEST(reflect_discovers_every_lane_member) {
    CHECK_EQ(L::count(), 7u);

    constexpr auto names = reflected_lane_names<Particle>();
    CHECK_EQ(names.size(), 7u);
    CHECK_EQ(names[0], std::string_view{"x"});
    CHECK_EQ(names[1], std::string_view{"y"});
    CHECK_EQ(names[2], std::string_view{"z"});
    CHECK_EQ(names[3], std::string_view{"vx"});
    CHECK_EQ(names[4], std::string_view{"vy"});
    CHECK_EQ(names[5], std::string_view{"vz"});
    CHECK_EQ(names[6], std::string_view{"life"});

    // Order is declaration order, which is what makes a reflected layout
    // deterministic across compilations.
    CHECK_EQ(F_x::name, std::string_view{"x"});
    CHECK_EQ(F_life::name, std::string_view{"life"});
}

VECTIS_TEST(reflect_filters_non_lane_members) {
    // char is one byte and bool is a predicate; both must be skipped, and the
    // members around them must keep their relative order.  Mixed declares six
    // members and four of them are lanes - the counting is the test.
    constexpr auto names = reflected_lane_names<Mixed>();
    CHECK_EQ(names.size(), 4u);
    CHECK_EQ(names[0], std::string_view{"a"});
    CHECK_EQ(names[1], std::string_view{"n"});
    CHECK_EQ(names[2], std::string_view{"d"});
    CHECK_EQ(names[3], std::string_view{"counter"});

    using LM = reflected_layout<Mixed>;
    static_assert(LM::count() == 4);
    static_assert(std::is_same_v<LM::field_at<0>::type, float>);
    static_assert(std::is_same_v<LM::field_at<1>::type, int>);
    static_assert(std::is_same_v<LM::field_at<2>::type, double>);
    static_assert(std::is_same_v<LM::field_at<3>::type, long>);
}

/// The filters that are not tests of size, and each of which used to be a hard
/// error inside a library header rather than a skip.
///
/// Asserting the count and the names, not merely that this compiles, is the
/// point: the failures these filters prevent were a `std::vector<const float>`
/// static assertion from <bits/stl_vector.h> and a "returning reference to
/// temporary" from the descriptor, neither of which names the member that
/// caused it.
VECTIS_TEST(reflect_filters_what_cannot_be_a_lane) {
    constexpr auto names = reflected_lane_names<Obstructed>();
    CHECK_EQ(names.size(), 3u);
    CHECK_EQ(names[0], std::string_view{"x"});
    CHECK_EQ(names[1], std::string_view{"n"});
    CHECK_EQ(names[2], std::string_view{"s"});

    using LO = reflected_layout<Obstructed>;
    static_assert(LO::count() == 3);
    static_assert(std::is_same_v<LO::field_at<0>::type, float>);
    static_assert(std::is_same_v<LO::field_at<1>::type, int>);
    static_assert(std::is_same_v<LO::field_at<2>::type, short>);

    // flags, frozen, x, v, n, ptr, s, c - in declaration order.
    Obstructed o{5u, 1.5f, 2.5f, 3, 4, nullptr, static_cast<short>(6), 'z'};

    soa_array<LO> soa(&o, 1);
    CHECK_EQ(soa.size(), 1u);
    CHECK_ULP(soa.data<LO::field_at<0>>()[0], 2.5f, 0);
    CHECK_EQ(soa.data<LO::field_at<1>>()[0], 4);
    CHECK_EQ(soa.data<LO::field_at<2>>()[0], short{6});

    // The write-back touches the lanes and nothing else: the bit-field and the
    // const member keep the values they were built with.
    Obstructed back{0u, 9.5f, 0.0f, 0, 0, nullptr, short{0}, '\0'};
    soa.to_aos(&back);
    CHECK_ULP(back.x, 2.5f, 0);
    CHECK_EQ(back.n, 4);
    CHECK_EQ(back.s, short{6});
    CHECK_EQ(back.flags, 0u);
    CHECK_ULP(back.frozen, 9.5f, 0);
    CHECK_EQ(back.c, '\0');
}

// ===========================================================================
// The conversion, and that it agrees with the annotated path
// ===========================================================================

VECTIS_TEST(reflect_aos_to_soa_roundtrip) {
    for (std::size_t n : {0u, 1u, 7u, 64u, 1000u}) {
        auto aos = make_aos();
        aos.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            aos[i].x = static_cast<float>(i) * 0.5f;
        }

        reflected_soa<Particle> soa(aos.data(), n);
        CHECK_EQ(soa.size(), n);
        CHECK_EQ(soa.field_bytes<F_x>(), n * sizeof(float));

        std::vector<Particle> back(n);
        soa.to_aos(back.data());
        for (std::size_t i = 0; i < n; ++i) {
            CHECK_ULP(back[i].x, aos[i].x, 0);
            CHECK_ULP(back[i].y, aos[i].y, 0);
            CHECK_ULP(back[i].z, aos[i].z, 0);
            CHECK_ULP(back[i].vx, aos[i].vx, 0);
            CHECK_ULP(back[i].vy, aos[i].vy, 0);
            CHECK_ULP(back[i].vz, aos[i].vz, 0);
            CHECK_ULP(back[i].life, aos[i].life, 0);
        }
    }
}

VECTIS_TEST(reflect_agrees_with_the_annotated_path) {
    // The two pathways - reflection and hand-written pointers to members -
    // produce *different descriptor types* with the same interface.  Asserting
    // type identity would be asserting something untrue; what has to hold is
    // that they describe the same layout, so a kernel written against one
    // behaves identically against the other.
    using Annotated = mem_layout<Particle, &Particle::x, &Particle::y,
                                 &Particle::z, &Particle::vx, &Particle::vy,
                                 &Particle::vz, &Particle::life>;

    static_assert(Annotated::count() == L::count());
    static_assert(std::is_same_v<typename Annotated::field_at<0>::type,
                                 typename L::field_at<0>::type>);
    static_assert(std::is_same_v<typename Annotated::field_at<1>::type,
                                 typename L::field_at<1>::type>);
    static_assert(std::is_same_v<typename Annotated::field_at<6>::type,
                                 typename L::field_at<6>::type>);
    static_assert(std::is_same_v<typename Annotated::owner_type, typename L::owner_type>);
    static_assert(std::is_same_v<typename L::field_at<3>::type, float>);
    static_assert(L::index_of<F_y>() == 1);

    // Structurally identical descriptors must not be *the same type* - if they
    // ever become one, this assertion is the reminder that the interface, not
    // the identity, is the contract.
    static_assert(!std::is_same_v<typename Annotated::field_at<0>,
                                  typename L::field_at<0>>);

    // And the observable consequence: both convert the same bytes the same way.
    auto aos = make_aos();
    reflected_soa<Particle> via_reflection(aos.data(), N);
    soa_array<Annotated>    via_pointers(aos.data(), N);
    for (std::size_t i = 0; i < N; ++i) {
        CHECK_ULP(via_reflection.data<F_x>()[i],
                  via_pointers.data<typename Annotated::field_at<0>>()[i], 0);
        CHECK_ULP(via_reflection.data<F_life>()[i],
                  via_pointers.data<typename Annotated::field_at<6>>()[i], 0);
    }
}

VECTIS_TEST(reflect_vector_kernel_over_reflected_storage) {
    // The point of the whole exercise: run a vectorised kernel over storage
    // whose layout was never declared.
    auto aos = make_aos();
    reflected_soa<Particle> soa(aos.data(), N);

    constexpr std::size_t VN = 16;
    const float dt = 0.25f;
    using V = basic_vec<float, VN>;
    const V vdt = V::set1(dt);

    const std::size_t done = soa.for_each_block<VN>([&](std::size_t i) {
        soa.store<F_x, VN>(i, soa.load<F_x, VN>(i) + soa.load<F_vx, VN>(i) * vdt);
        soa.store<F_y, VN>(i, soa.load<F_y, VN>(i) + soa.load<F_vy, VN>(i) * vdt);
        soa.store<F_z, VN>(i, soa.load<F_z, VN>(i) + soa.load<F_vz, VN>(i) * vdt);
    });
    CHECK_EQ(done, N);

    std::vector<Particle> got(N);
    soa.to_aos(got.data());
    for (std::size_t i = 0; i < N; ++i) {
        CHECK_ULP(got[i].x, aos[i].x + aos[i].vx * dt, 0);
        CHECK_ULP(got[i].y, aos[i].y + aos[i].vy * dt, 0);
        CHECK_ULP(got[i].z, aos[i].z + aos[i].vz * dt, 0);
    }
}

#else

VECTIS_TEST(reflect_not_available) {
    vtest::skip("this compiler has no reflection "
                "(needs GCC 16 with -freflection)");
}

#endif
