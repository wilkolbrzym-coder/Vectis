// ===========================================================================
// Vectis - compile-time SIMD engine
// core/concepts.hpp - the vocabulary the engine is written against
// ===========================================================================
//
// These concepts are load-bearing, not decorative.  Every kernel in the library
// is constrained by them, so a misuse produces a one-line diagnostic at the
// call site instead of a 400-line template backtrace.
//
//   Vectorizable<T>    - T is a scalar a SIMD lane can hold
//   SimdAbi<A>         - A names a register-file strategy
//   SimdVec<V>         - V behaves like a fixed-width vector of lanes
//   SimdMask<M, V>     - M is the mask type paired with V
//   Field<D, T>        - D describes one member of struct T  (drives AoS -> SoA)
//   SoaLayout<L, T>    - L is a field bundle over T
//   UnaryLaneOp / BinaryLaneOp - elementwise kernel shapes
//
// ===========================================================================
#pragma once

#include "config.hpp"

#include <concepts>
#include <cstddef>
#include <string_view>
#include <type_traits>

namespace vectis {

// --------------------------------------------------------------- scalar lanes

/// Scalar types a SIMD lane may hold.
///
/// Deliberately narrower than std::arithmetic: `bool` is excluded because it is
/// a *predicate*, not a lane (it is what a mask holds), and so are the extended
/// floating-point types whose ABI support varies per target.
template <class T>
concept Vectorizable =
    std::is_arithmetic_v<T> &&
    !std::is_same_v<std::remove_cv_t<T>, bool> &&
    sizeof(T) >= 2;

/// A lane type whose arithmetic a floating-point kernel can rely on.
template <class T>
concept FloatLane = std::floating_point<T> && (sizeof(T) == 4 || sizeof(T) == 8);

/// A lane type an integer kernel can do wrapping arithmetic on.
template <class T>
concept IntLane = std::integral<T> &&
                  !std::is_same_v<std::remove_cv_t<T>, bool>;

// ------------------------------------------------------------------ ABI tags

/// Tag type naming a register-file strategy (scalar / sse / avx2 / avx512).
template <class A>
concept SimdAbi = std::is_empty_v<A> && requires {
    { A::isa } -> std::convertible_to<isa_level>;
};

// ------------------------------------------------------------ vector protocol

/// A SIMD vector: a fixed lane count over a Vectorizable element type, carried
/// by the register storage its ABI tag implies.
///
/// `lanes` is the architectural lane count the *programmer* asked for.  It is
/// not necessarily the register width: `vec<float, 16>` is two YMM registers on
/// AVX2 and one ZMM on AVX-512, and that is the whole point of the engine.
template <class V>
concept SimdVec = requires {
    typename V::value_type;
    typename V::abi_type;
    V::lanes;
    { V::lane_count() } -> std::convertible_to<std::size_t>;
} &&
    Vectorizable<typename V::value_type> &&
    SimdAbi<typename V::abi_type> &&
    (V::lanes > 0);

/// The mask type paired with a SimdVec: one predicate per lane.
template <class M, class V>
concept SimdMask = SimdVec<V> && requires {
    typename M::value_type;
    M::lanes;
} &&
    std::same_as<typename M::value_type, bool> &&
    (M::lanes == V::lanes);

/// Two vectors share an element type and a lane count, so they can be combined.
template <class A, class B>
concept SameShape = SimdVec<A> && SimdVec<B> &&
    std::same_as<typename A::value_type, typename B::value_type> &&
    (A::lanes == B::lanes);

/// A vector whose lanes are the given element type.
template <class V, class T>
concept VecOf = SimdVec<V> &&
    std::same_as<std::remove_cv_t<typename V::value_type>,
                 std::remove_cv_t<T>>;

// ------------------------------------------------------- struct layout (SoA)

/// Describes one member of a user structure so the engine can split an array of
/// those structures into parallel arrays - the AoS -> SoA transformation.
///
/// A descriptor exposes:
///   * `owner_type`      - the struct the member belongs to
///   * `type`            - the member's type
///   * `get(const T&)`   - read the member
///   * `set(T&, type)`   - write the member
///
/// In vectis/soa these are generated from a pointer-to-member used as a
/// non-type template parameter, which carries both the owner and the member
/// type and therefore needs no macro or code generation.  Under C++26
/// reflection (P2996) the same interface is synthesised from
/// `nonstatic_data_members_of` instead.
template <class D, class T>
concept Field =
    requires(const T& cobj, T& obj, typename D::type v) {
        typename D::owner_type;
        { D::get(cobj) } -> std::convertible_to<typename D::type>;
        D::set(obj, v);
    } &&
    std::same_as<typename D::owner_type, T> &&
    Vectorizable<typename D::type>;

/// A bundle of Fields over a common struct type.
///
/// The `T` parameter is load-bearing, not a formality: a layout names its owner
/// through `L::owner_type` (see vectis::layout in soa/soa.hpp), and the two are
/// required to agree here.  Without that, a concept whose whole purpose is to
/// say "L is a field bundle over T" was satisfied by any `L` that happened to
/// have a `count()` - including one describing a completely different struct.
///
/// The per-field check is already enforced where the fields are declared:
/// `layout` static_asserts `(FieldOf<Fields, Owner> && ...)` on itself, so a
/// layout that exists at all has fields of the right kind.
template <class L, class T>
concept SoaLayout = requires {
    typename L::owner_type;
    { L::count() } -> std::convertible_to<std::size_t>;
} &&
    std::same_as<typename L::owner_type, T> &&
    (L::count() > 0);

// ------------------------------------------------------------- kernel shapes

/// `out[i] = f(in[i])`.
template <class F, class T>
concept UnaryLaneOp = Vectorizable<T> &&
    requires(F f, T x) { { f(x) } -> std::convertible_to<T>; };

/// `out[i] = f(a[i], b[i])`.
template <class F, class T>
concept BinaryLaneOp = Vectorizable<T> &&
    requires(F f, T x, T y) { { f(x, y) } -> std::convertible_to<T>; };

/// A callable usable as a per-lane predicate.
template <class P, class T>
concept LanePredicate = Vectorizable<T> &&
    requires(P p, T x) { { p(x) } -> std::convertible_to<bool>; };

} // namespace vectis
