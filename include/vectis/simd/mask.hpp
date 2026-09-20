// ===========================================================================
// Vectis - compile-time SIMD engine
// simd/mask.hpp - per-lane predicates
// ===========================================================================
//
// AVX-512 has first-class mask registers (k0..k7); AVX2 and SSE do not, and
// express a predicate as an all-ones/all-zeros vector instead.  Both shape a
// `mask_reg`, and both are hidden behind this class so a kernel never cares.
//
// The portable surface - test/all/any/none/count/bits - is what kernels should
// use.  bits() is the escape hatch: at N <= 64 lanes a whole vector's predicate
// fits in a single integer, which is what branchy code (ray-box slab tests,
// AABB culling) actually wants to consume.
//
// ===========================================================================
#pragma once

#include "backend.hpp"
#include "backends.hpp"
#include "detail.hpp"

#include <array>
#include <bit>
#include <cstdint>

namespace vectis {

template <class T, std::size_t N, class Abi = native_abi>
    requires ValidLaneCount<T, Abi, N>
class basic_mask {
public:
    using value_type   = bool;
    using abi_type     = Abi;
    using backend_type = backend<T, Abi>;
    using mask_reg     = typename backend_type::mask_reg;

    /// All N lane bits set, for masking padding out of the packed predicate.
    [[nodiscard]] static constexpr std::uint64_t lane_mask() noexcept {
        return (N >= 64) ? ~std::uint64_t{0} : ((std::uint64_t{1} << N) - 1);
    }

    static constexpr std::size_t lanes     = N;
    static constexpr std::size_t reg_lanes = backend_type::lanes;
    /// Same rounding as basic_vec: a lane count that does not fill its last
    /// register still occupies a whole one.
    static constexpr std::size_t num_regs  = detail::div_ceil<N, reg_lanes>;

    static_assert(num_regs >= 1);
    static_assert(N <= 64, "Vectis masks are limited to 64 lanes: bits() returns "
                           "a single 64-bit word, which is the only portable "
                           "predicate carrier across tiers");

    [[nodiscard]] static constexpr std::size_t lane_count() noexcept { return N; }
    [[nodiscard]] static constexpr std::size_t size() noexcept { return N; }

    /// All lanes false.
    constexpr basic_mask() noexcept {
        for (std::size_t i = 0; i < num_regs; ++i) {
            m_[i] = backend_type::mask_false();
        }
    }

    explicit constexpr basic_mask(const mask_reg (&regs)[num_regs]) noexcept {
        for (std::size_t i = 0; i < num_regs; ++i) m_[i] = regs[i];
    }

    /// Build a mask by asking `f(register_index)` for each register.  This is
    /// how a comparison gets its result without anyone naming an array of the
    /// register type in a template argument position.
    template <class F>
    [[nodiscard]] static basic_mask generate(F&& f) noexcept {
        basic_mask out;
        for (std::size_t i = 0; i < num_regs; ++i) out.m_[i] = f(i);
        return out;
    }

    // ------------------------------------------------------------- factories
    // NOTE: there is no static none().  A default-constructed mask is already
    // all-false, and `all()`/`any()`/`none()` are the *observers* below - a
    // static factory of the same name cannot coexist with them.

    /// Every lane true.
    [[nodiscard]] static basic_mask full() noexcept {
        basic_mask out;
        for (std::size_t i = 0; i < num_regs; ++i) {
            out.m_[i] = backend_type::mask_true();
        }
        return out;
    }

    [[nodiscard]] static basic_mask from_bits(std::uint64_t b) noexcept {
        basic_mask out;
        for (std::size_t i = 0; i < num_regs; ++i) {
            const auto shift = static_cast<unsigned>(i * reg_lanes);
            const auto chunk = (reg_lanes >= 64)
                                   ? b
                                   : ((b >> shift) & ((std::uint64_t{1} << reg_lanes) - 1));
            out.m_[i] = backend_type::mask_from_bits(chunk);
        }
        return out;
    }

    /// A mask with exactly lane `i` set.  Out-of-range lanes give an empty mask.
    [[nodiscard]] static basic_mask from_lane(std::size_t i) noexcept {
        if (i >= N) return basic_mask{};
        return from_bits(std::uint64_t{1} << i);
    }

    // -------------------------------------------------------------- observers
    [[nodiscard]] bool test(std::size_t lane) const noexcept {
        if (lane >= N) return false;
        const std::size_t r = lane / reg_lanes;
        const std::size_t l = lane % reg_lanes;
        return backend_type::mask_test(m_[r], l);
    }

    /// Lane `i` as a bool.  Alias for test(), for readable range-for code.
    [[nodiscard]] bool operator[](std::size_t lane) const noexcept {
        return test(lane);
    }

    /// The N predicates packed into one word, bit i for lane i.
    ///
    /// Masked to N bits on purpose: a partially-filled final register compares
    /// its padding lanes too, and those bits must never reach the caller - they
    /// would corrupt count(), all() and find_first().
    [[nodiscard]] std::uint64_t bits() const noexcept {
        std::uint64_t out = 0;
        for (std::size_t i = 0; i < num_regs; ++i) {
            out |= backend_type::mask_bits(m_[i]) << (i * reg_lanes);
        }
        return out & lane_mask();
    }

    [[nodiscard]] bool none() const noexcept { return bits() == 0; }

    [[nodiscard]] bool any() const noexcept { return !none(); }

    [[nodiscard]] bool all() const noexcept { return bits() == lane_mask(); }

    /// Number of true lanes.
    ///
    /// Goes through bits() rather than summing the raw registers: the padding
    /// lanes of a partial final register hold a real predicate too (they
    /// compare equal, because both sides were zero-filled by the load), and
    /// counting them would report more true lanes than the vector has.
    [[nodiscard]] std::size_t count() const noexcept {
        return static_cast<std::size_t>(std::popcount(bits()));
    }

    [[nodiscard]] std::array<bool, N> to_array() const noexcept {
        std::array<bool, N> out{};
        for (std::size_t i = 0; i < N; ++i) out[i] = test(i);
        return out;
    }

    /// Index of the lowest set lane, or N when empty.  This is the primitive
    /// behind "process only the lanes that survived a test".
    [[nodiscard]] std::size_t find_first() const noexcept
        requires (N <= 64) {
        const std::uint64_t b = bits();
        return b == 0 ? N : static_cast<std::size_t>(std::countr_zero(b));
    }

    /// Iterate set lanes: `for (auto lane : mask.lanes()) { ... }`.
    class lane_iterator {
    public:
        lane_iterator(std::uint64_t bits, std::size_t pos) noexcept
            : bits_(bits), pos_(pos) {}
        [[nodiscard]] std::size_t operator*() const noexcept { return pos_; }
        lane_iterator& operator++() noexcept {
            bits_ &= bits_ - 1;                    // clear lowest set bit
            pos_ = bits_ == 0 ? N : static_cast<std::size_t>(std::countr_zero(bits_));
            return *this;
        }
        [[nodiscard]] bool operator!=(const lane_iterator& o) const noexcept {
            return pos_ != o.pos_;
        }
    private:
        std::uint64_t bits_;
        std::size_t pos_;
    };

    struct lane_range {
        std::uint64_t bits;
        [[nodiscard]] lane_iterator begin() const noexcept {
            return {bits, bits == 0 ? N : static_cast<std::size_t>(std::countr_zero(bits))};
        }
        [[nodiscard]] lane_iterator end() const noexcept { return {0, N}; }
    };

    /// The set lanes, in ascending order.
    [[nodiscard]] lane_range lanes_set() const noexcept { return {bits()}; }

    // -------------------------------------------------------------- operators
    [[nodiscard]] friend basic_mask operator&(const basic_mask& a, const basic_mask& b) noexcept {
        basic_mask out;
        for (std::size_t i = 0; i < num_regs; ++i) {
            out.m_[i] = backend_type::mask_and(a.m_[i], b.m_[i]);
        }
        return out;
    }
    [[nodiscard]] friend basic_mask operator|(const basic_mask& a, const basic_mask& b) noexcept {
        basic_mask out;
        for (std::size_t i = 0; i < num_regs; ++i) {
            out.m_[i] = backend_type::mask_or(a.m_[i], b.m_[i]);
        }
        return out;
    }
    [[nodiscard]] friend basic_mask operator^(const basic_mask& a, const basic_mask& b) noexcept {
        basic_mask out;
        for (std::size_t i = 0; i < num_regs; ++i) {
            out.m_[i] = backend_type::mask_xor(a.m_[i], b.m_[i]);
        }
        return out;
    }
    [[nodiscard]] basic_mask operator~() const noexcept {
        basic_mask out;
        for (std::size_t i = 0; i < num_regs; ++i) {
            out.m_[i] = backend_type::mask_not(m_[i]);
        }
        return out;
    }

    /// Equality over the N observable lanes.
    ///
    /// Deliberately routed through bits() rather than comparing the raw
    /// registers: the padding lanes of a partial final register hold a real
    /// predicate, so two masks that agree on every lane the caller can see -
    /// and on which test(), count(), all(), any() and bits() all agree - can
    /// still differ there.  Comparing the packed, lane-masked predicate is the
    /// only spelling consistent with every other observer in this class.
    [[nodiscard]] friend bool operator==(const basic_mask& a, const basic_mask& b) noexcept {
        return a.bits() == b.bits();
    }
    [[nodiscard]] friend bool operator!=(const basic_mask& a, const basic_mask& b) noexcept {
        return !(a == b);
    }

    [[nodiscard]] const mask_reg (&raw() const noexcept)[num_regs] { return m_; }

private:
    /// Plain C array for the same reason as basic_vec::r_ - see the note there.
    mask_reg m_[num_regs];
};

} // namespace vectis
