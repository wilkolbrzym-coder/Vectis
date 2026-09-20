#pragma once

#include "../core/concepts.hpp"
#include "../simd/reduce.hpp"
#include "../simd/vec.hpp"

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace vectis {

// ===========================================================================
// Field descriptors
// ===========================================================================
//
// A field is named by a pointer to member, used as a non-type template
// parameter:
//
//     using F_x = vectis::field_of<&Particle::x>;
//
// That one expression carries everything a descriptor needs - the owning type
// and the member's type are both deduced from `float Particle::*` - so there is
// no macro here, nothing to keep in sync with the struct, and it composes the
// way a type should.
//
// This is deliberately the same information C++26 reflection would hand over
// (`nonstatic_data_members_of(^^Particle)` gives members, and a pointer to each
// is a `std::meta::info` short of what is written below).  When P2996 ships,
// `field_of` can be generated instead of spelled, and nothing downstream
// changes, because the container only sees this interface.
//
// The one thing a pointer cannot carry is the member's *name*, which is why the
// descriptor has no `name` member.  It is a diagnostic nicety, not a capability.

/// Descriptor for one member of a struct.
///
/// Primary template is intentionally undefined: `field_of<X>` for anything that
/// is not a pointer to a data member is a compile error naming the argument,
/// not a silently empty descriptor.
template <auto MemberPtr>
struct field_of;

template <class Owner, class T, T Owner::* Ptr>
struct field_of<Ptr> {
    using owner_type = Owner;
    using type       = T;

    static constexpr T& get(Owner& o) noexcept { return o.*Ptr; }
    static constexpr const T& get(const Owner& o) noexcept { return o.*Ptr; }
    static constexpr void set(Owner& o, T v) noexcept { o.*Ptr = v; }
};

/// True when `D` is a field descriptor usable over `Owner`.
template <class D, class Owner>
concept FieldOf = requires(const Owner& c, Owner& m, typename D::type v) {
    typename D::owner_type;
    requires std::same_as<typename D::owner_type, Owner>;
    requires Vectorizable<typename D::type>;
    { D::get(c) } -> std::convertible_to<typename D::type>;
    D::set(m, v);
};

// ===========================================================================
// Layout: an ordered list of fields over one struct
// ===========================================================================

/// An ordered list of field descriptors over one struct.
///
/// Keyed on descriptor *types* rather than on pointers-to-member, because that
/// is the interface both producers can satisfy: a hand-written descriptor
/// (`field_of<&Particle::x>`) and one generated from a reflection
/// (`reflected_field<^^Particle::x>`) are both types, and the container cannot
/// tell them apart.  `mem_layout` below is the pointer-to-member spelling.
///
///     using L = layout<Particle, field_of<&Particle::x>, field_of<&Particle::y>>;
template <class Owner, class... Fields>
struct layout {
    using owner_type = Owner;
    using fields     = std::tuple<Fields...>;

    static constexpr std::size_t count() noexcept { return sizeof...(Fields); }

    static_assert(count() > 0, "a layout needs at least one field");
    static_assert((FieldOf<Fields, Owner> && ...),
                  "every field in a layout must be a vectorisable data member "
                  "of the same struct; nested structs are not lanes");

    /// The tuple of per-field arrays this layout implies.
    using storage_type = std::tuple<std::vector<typename Fields::type>...>;

    template <std::size_t I>
    using field_at = std::tuple_element_t<I, fields>;

    /// Position of a field in the layout, or count() when it is not in it.
    ///
    /// Index lookup rather than `std::get<Field>`: the fields of a layout very
    /// often share a type - x, y and z are all float - and a by-type get would
    /// be ambiguous.
    template <class F>
    static constexpr std::size_t index_of() noexcept {
        constexpr bool matches[] = {std::is_same_v<F, Fields>...};
        for (std::size_t i = 0; i < count(); ++i) {
            if (matches[i]) return i;
        }
        return count();
    }

    /// Apply a callable to each field descriptor, in declaration order:
    ///
    ///     Layout::for_each_field([&]<class F>() { ... });
    template <class Fn>
    static constexpr void for_each_field(Fn&& fn) {
        (fn.template operator()<Fields>(), ...);
    }
};

/// The same layout, spelled with pointers to members:
///
///     using L = mem_layout<Particle, &Particle::x, &Particle::y>;
///
/// Pure sugar: it names the descriptor types `field_of` would have produced.
/// Kept because it is the shortest thing to write by hand, and because it is
/// the only spelling available before C++26 reflection.
template <class Owner, auto... Members>
using mem_layout = layout<Owner, field_of<Members>...>;

// ===========================================================================
// SoA container
// ===========================================================================

/// A structure-of-arrays: one contiguous array per field, all the same length.
template <class Layout>
class soa_array {
public:
    using layout_type = Layout;
    using owner_type  = typename Layout::owner_type;

    soa_array() = default;

    explicit soa_array(std::size_t n) { resize(n); }

    /// Build from an existing array of structs: the AoS -> SoA conversion.
    soa_array(const owner_type* src, std::size_t n) { assign(src, n); }

    // ------------------------------------------------------- special members
    //
    // Written out rather than defaulted, because the defaulted move constructor
    // and move assignment would leave `size_` describing a container whose
    // arrays had just been emptied - the one invariant this class has is
    // `size_ == arrays_[i].size()` for every field, and a moved-from object
    // would then report the pre-move size through size(), empty() and
    // field_bytes() while data<F>() returned nullptr.  Anything reading it -
    // to_aos, load<F,N>, for_each_block - would dereference that null.
    // The moved-from object is left empty and consistent, as the standard
    // containers are.
    soa_array(const soa_array&) = default;
    soa_array(soa_array&& other) noexcept
        : arrays_(std::move(other.arrays_)),
          size_(std::exchange(other.size_, 0)) {}

    soa_array& operator=(const soa_array&) = default;
    soa_array& operator=(soa_array&& other) noexcept {
        arrays_ = std::move(other.arrays_);
        size_   = std::exchange(other.size_, 0);
        return *this;
    }

    void swap(soa_array& other) noexcept {
        using std::swap;
        swap(arrays_, other.arrays_);
        swap(size_, other.size_);
    }
    friend void swap(soa_array& a, soa_array& b) noexcept { a.swap(b); }

    /// Grow or shrink every field to `n`.
    ///
    /// `size_` is written last, so an allocation failure part-way through
    /// leaves every array at least as long as `size_` claims and no read can
    /// leave the storage - the container stays usable, just under-resized.
    void resize(std::size_t n) {
        Layout::for_each_field([&]<class F>() {
            std::get<Layout::template index_of<F>()>(arrays_).resize(n);
        });
        size_ = n;
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    /// Contiguous storage for one field.  Public and cheap: a contiguous
    /// per-field array is the whole point of the layout, and hiding it would
    /// only force callers back through a gather.
    template <class F>
    [[nodiscard]] typename F::type* data() noexcept {
        return std::get<Layout::template index_of<F>()>(arrays_).data();
    }
    template <class F>
    [[nodiscard]] const typename F::type* data() const noexcept {
        return std::get<Layout::template index_of<F>()>(arrays_).data();
    }

    /// Bytes held for one field, for tests and diagnostics.
    template <class F>
    [[nodiscard]] std::size_t field_bytes() const noexcept {
        return std::get<Layout::template index_of<F>()>(arrays_).size() *
               sizeof(typename F::type);
    }

    // ---------------------------------------------------------------- convert

    /// AoS -> SoA.  One pass per field.
    void assign(const owner_type* src, std::size_t n) {
        resize(n);
        Layout::for_each_field([&]<class F>() {
            auto* dst = data<F>();
            for (std::size_t i = 0; i < n; ++i) dst[i] = F::get(src[i]);
        });
    }

    /// SoA -> AoS.
    void to_aos(owner_type* dst) const {
        Layout::for_each_field([&]<class F>() {
            const auto* src = data<F>();
            for (std::size_t i = 0; i < size_; ++i) F::set(dst[i], src[i]);
        });
    }

    // ----------------------------------------------------------- vector access

    /// Load N lanes of one field, starting at element `i`.
    template <class F, std::size_t N>
    [[nodiscard]] basic_vec<typename F::type, N> load(std::size_t i) const
        noexcept {
        return basic_vec<typename F::type, N>::load(data<F>() + i);
    }

    /// Store N lanes of one field, starting at element `i`.
    template <class F, std::size_t N>
    void store(std::size_t i, const basic_vec<typename F::type, N>& v) noexcept {
        v.store(data<F>() + i);
    }

    /// Visit every block of exactly N elements.
    ///
    /// The ragged tail is deliberately *not* handled here: only the caller
    /// knows what its kernel does, and a library that guessed would either
    /// duplicate every kernel or do something subtly different from it.  The
    /// loop count is returned so the caller can pick up from there.
    ///
    ///     std::size_t done = arr.for_each_block<8>([&](std::size_t i) { ... });
    ///     for (; done < arr.size(); ++done) { /* scalar tail */ }
    ///
    /// N must be positive: `for_each_block<0>` would advance by zero and never
    /// terminate, and the compiler can say so here rather than leaving the
    /// caller with a hang to diagnose.
    template <std::size_t N, class Fn>
    std::size_t for_each_block(Fn&& fn) {
        static_assert(N > 0, "for_each_block needs a block size of at least one");
        std::size_t i = 0;
        for (; i + N <= size_; i += N) fn(i);
        return i;
    }

    template <std::size_t N, class Fn>
    std::size_t for_each_block(Fn&& fn) const {
        static_assert(N > 0, "for_each_block needs a block size of at least one");
        std::size_t i = 0;
        for (; i + N <= size_; i += N) fn(i);
        return i;
    }

private:
    typename Layout::storage_type arrays_;
    std::size_t size_ = 0;
};

} // namespace vectis
