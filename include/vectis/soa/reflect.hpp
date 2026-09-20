// ===========================================================================
// Vectis - AoS to SoA with real reflection
// ===========================================================================
//
// This is the file the project was always aiming at, and it is short because
// reflection finally does the work.
//
// The hand-written path in soa.hpp needs one pointer-to-member per field:
//
//     using L = mem_layout<Particle, &Particle::x, &Particle::y, ...>;
//
// That is a declaration per member, kept in sync by hand, and it silently goes
// wrong the day somebody adds a field.  With P2996 the compiler reads the
// struct itself:
//
//     using L = reflected_layout<Particle>;
//     soa_array<L> particles(aos, n);       // no field list anywhere
//
// and, with the deduction guide at the bottom, not even that:
//
//     soa_array particles(aos, n);
//
// The layout it produces is an ordinary `layout<Owner, Field...>` full of
// ordinary descriptors, so everything downstream - the container, the kernels,
// the tests - is the code that was already there.  Reflection is confined to
// this file, which is also why the pointer-to-member path is kept: it is the
// C++20 answer to the same question, and it still works.
//
// Caveat, stated because it is easy to trip over: `substitute` on a concept is
// not used here to filter members, because the result of substituting into a
// concept-id is not reliably extractable as a value in this implementation.
// The filter is written out with the type predicates instead, which are
// equivalent for Vectorizable and demonstrably work.
//
// ===========================================================================
#pragma once

#include "../core/config.hpp"

#if defined(VECTIS_HAS_REFLECTION)

#include "../core/concepts.hpp"
#include "soa.hpp"

#include <cstddef>
#include <meta>
#include <string_view>
#include <array>
#include <utility>
#include <vector>

namespace vectis {

/// The data members of `type` that can occupy a vector lane.
///
/// Public, non-static, non-bit-field, unqualified, arithmetic, not bool, at
/// least two bytes - which is what the Vectorizable concept means, spelled with
/// reflection predicates because a concept cannot be queried directly from a
/// consteval function in this implementation.
///
/// Members that fail the filter are skipped rather than rejected, so a struct
/// with a name string or a flags enum alongside its numbers still works: the
/// numbers become lanes and the rest is ignored.
///
/// Three of those checks are not obvious and are each a real defect when
/// missing, because the mistake surfaces as an error inside a header the caller
/// never included rather than as a message about their struct:
///
///   * `is_bit_field` - a named bit-field is a data member and its type is
///     arithmetic, so it passes every other check.  It cannot be a lane: the
///     container stores `std::vector<unsigned>` and the descriptor's
///     `get()`/`set()` cannot take a reference to a bit-field.
///   * `is_const`/`is_volatile` - `is_arithmetic_type(const float)` is true, so
///     a `const float` member would become `std::vector<const float>`, which
///     libstdc++ rejects with "std::vector must have a non-const,
///     non-volatile value_type" pointing into <bits/stl_vector.h>.
///   * `is_public` after the access-context filter - `access_context::current()`
///     has already excluded inaccessible members in this implementation, so
///     this one is belt and braces rather than load-bearing.
[[nodiscard]] consteval std::vector<std::meta::info>
lane_members_of(std::meta::info type) {
    std::vector<std::meta::info> out;
    for (auto m : std::meta::nonstatic_data_members_of(
             type, std::meta::access_context::current())) {
        if (!std::meta::is_public(m)) continue;
        if (std::meta::is_bit_field(m)) continue;
        const auto t = std::meta::type_of(m);
        if (std::meta::is_const(t) || std::meta::is_volatile(t)) continue;
        if (!std::meta::is_arithmetic_type(t)) continue;
        if (std::meta::is_same_type(t, ^^bool)) continue;
        if (std::meta::size_of(t) < 2) continue;
        out.push_back(m);
    }
    return out;
}

/// A field descriptor built from a reflected data member.
///
/// Same interface as the hand-written descriptor in soa.hpp - the container
/// cannot tell them apart, which is the point.
template <std::meta::info Member>
struct reflected_field {
    using owner_type = [: std::meta::parent_of(Member) :];
    using type       = [: std::meta::type_of(Member) :];
    static constexpr std::meta::info member = Member;

    /// The member's name, for diagnostics.  The one thing a pointer-to-member
    /// descriptor could not carry, and the reason this path is worth having
    /// even where the other one is available.
    static constexpr std::string_view name = std::meta::identifier_of(Member);

    static constexpr type& get(owner_type& o) noexcept { return o.[: Member :]; }
    static constexpr const type& get(const owner_type& o) noexcept {
        return o.[: Member :];
    }
    static constexpr void set(owner_type& o, type v) noexcept {
        o.[: Member :] = v;
    }
};

/// How many lanes a struct has.  A consteval function rather than
/// `define_static_array(...).size()`, because this implementation returns a
/// span of *dynamic* extent, whose size is therefore not a constant expression.
[[nodiscard]] consteval std::size_t lane_member_count(std::meta::info type) {
    return lane_members_of(type).size();
}

/// The I-th lane member of T, as a constant expression usable as a template
/// argument.  Going through a consteval function sidesteps the span-extent
/// problem above entirely.
template <class T, std::size_t I>
[[nodiscard]] consteval std::meta::info lane_member() {
    const auto members = lane_members_of(^^T);
    return members[I];
}

namespace detail {

/// Expands the reflected member list into `layout<T, reflected_field<...>...>`.
template <class T>
struct reflected_layout_builder {
    /// The two ways this reflection is not the whole struct, stated here so
    /// they are compile errors about the caller's type rather than silent
    /// omissions that show up as wrong numbers later.
    ///
    /// `nonstatic_data_members_of` returns a type's *own* members only, so an
    /// inherited field would simply not become a lane - and it cannot be added
    /// by hand either, because `parent_of` on it names the base class, so the
    /// descriptor would not satisfy FieldOf over the derived type.  Rejecting
    /// the shape is the honest answer until bases are handled properly.
    static_assert(std::meta::bases_of(^^T,
                      std::meta::access_context::current()).empty(),
        "vectis::reflected_layout: a type with a base class is not supported. "
        "Reflection sees only a type's own data members, so an inherited field "
        "would be silently missing from the layout. Declare the inherited "
        "members in the derived type, or use mem_layout<> with an explicit "
        "list of pointers to members.");

    static_assert(lane_member_count(^^T) > 0,
        "vectis::reflected_layout: this type has no member that can be a vector "
        "lane. A lane has to be a public, non-static, unqualified, arithmetic "
        "data member of at least two bytes - so no bool, no char, no pointer, "
        "no string, no bit-field, no reference, and nothing inherited.");

    template <std::size_t... Is>
    static auto build(std::index_sequence<Is...>)
        -> layout<T, reflected_field<lane_member<T, Is>()>...>;

    using type = decltype(build(
        std::make_index_sequence<lane_member_count(^^T)>{}));
};

} // namespace detail

/// The SoA layout of a struct, read from the struct itself.
///
///     struct Particle { float x, y, z; };
///     using L = vectis::reflected_layout<Particle>;   // that is all
///
/// Fails to compile with a readable message when the struct has no vectorisable
/// members, rather than producing an empty layout that fails somewhere deep in
/// the container.
template <class T>
using reflected_layout = typename detail::reflected_layout_builder<T>::type;

/// A SoA container over a struct, with no field list.
template <class T>
using reflected_soa = soa_array<reflected_layout<T>>;

/// Names of the members that became lanes, in layout order.  Diagnostics only -
/// no run-time cost, and the only way to see what reflection decided.
template <class T>
[[nodiscard]] constexpr auto reflected_lane_names() {
    // Expanded through an index sequence, not a runtime loop: lane_member is a
    // template, so the index has to be a constant expression.
    return []<std::size_t... Is>(std::index_sequence<Is...>) {
        return std::array<std::string_view, sizeof...(Is)>{
            std::meta::identifier_of(lane_member<T, Is>())...
        };
    }(std::make_index_sequence<lane_member_count(^^T)>{});
}

} // namespace vectis

#endif // VECTIS_HAS_REFLECTION
