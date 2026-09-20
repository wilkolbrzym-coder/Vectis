// ===========================================================================
// Vectis - compile-time SIMD engine
// simd/vec.hpp - the vector type the whole engine is built on
// ===========================================================================
//
// `basic_vec<T, N, Abi>` is N lanes of T, carried by as many registers as Abi
// requires.  That last part is the point of the design:
//
//     using v16 = vectis::basic_vec<float, 16>;   // one line, three machines
//
//     on AVX-512  ->  one  ZMM   register
//     on AVX2     ->  two  YMM   registers
//     on scalar   ->  sixteen floats
//
// and every kernel written against it works unchanged on all three.  The wider
// vector is not a different type with different code; it is the same code with
// a longer fold, which the compiler unrolls because num_regs is constexpr.
//
// Comparison operators return a mask, not a bool - `a < b` is a per-lane
// predicate, exactly as in std::simd.  Use `all()`, `any()` or `bits()` to
// collapse it.
//
// ===========================================================================
#pragma once

#include "backend.hpp"
#include "backends.hpp"
#include "mask.hpp"

#include <array>
#include <cstdint>

namespace vectis {

template <class T, std::size_t N, class Abi = native_abi>
    requires ValidLaneCount<T, Abi, N>
class basic_vec {
public:
    using value_type   = T;
    using abi_type     = Abi;
    using backend_type = backend<T, Abi>;
    using reg_type     = typename backend_type::reg;
    using mask_type    = basic_mask<T, N, Abi>;

    static constexpr std::size_t lanes     = N;
    static constexpr std::size_t reg_lanes = backend_type::lanes;
    /// Whole registers needed to carry N lanes.  The last one may be partial.
    static constexpr std::size_t num_regs  = detail::div_ceil<N, reg_lanes>;
    /// Registers that are completely filled with live lanes.
    static constexpr std::size_t full_regs = N / reg_lanes;
    /// Live lanes in the final register; 0 when the vector fills it exactly.
    static constexpr std::size_t tail_lanes = N % reg_lanes;

    static_assert(num_regs >= 1);

    [[nodiscard]] static constexpr std::size_t lane_count() noexcept { return N; }
    [[nodiscard]] static constexpr std::size_t size() noexcept { return N; }

    /// Zero-initialised, unlike a raw SIMD register.  Deterministic default
    /// construction is worth one instruction and removes a whole class of
    /// "worked in debug, garbage in release" bugs.
    ///
    /// Written out rather than `= default` because the storage is a plain C
    /// array (see the note on `r_` below), which `= default` would leave
    /// uninitialised.
    constexpr basic_vec() noexcept {
        for (std::size_t i = 0; i < num_regs; ++i) r_[i] = backend_type::zero();
    }

    /// Wrap a single register.  Only for vectors that are exactly one register
    /// wide; it exists so intrinsics code can interoperate without a round trip
    /// through memory.
    explicit constexpr basic_vec(reg_type r) noexcept
        requires (num_regs == 1) {
        r_[0] = r;
    }

    /// Wrap an array of registers, one per register the vector occupies.
    explicit constexpr basic_vec(const reg_type (&regs)[num_regs]) noexcept {
        for (std::size_t i = 0; i < num_regs; ++i) r_[i] = regs[i];
    }

    /// Per-lane construction: `f32x4 v{1.0f, 2.0f, 3.0f, 4.0f}`.
    template <class... Us>
        requires (sizeof...(Us) == N) && (N > 1) &&
                 (std::convertible_to<Us, T> && ...)
    constexpr basic_vec(Us... vals) noexcept {
        const std::array<T, N> tmp{static_cast<T>(vals)...};
        for (std::size_t i = 0; i < num_regs; ++i) {
            r_[i] = backend_type::load(tmp.data() + i * reg_lanes);
        }
    }

    // ------------------------------------------------------------- factories
    [[nodiscard]] static basic_vec zero() noexcept {
        basic_vec out;
        for (std::size_t i = 0; i < num_regs; ++i) {
            out.r_[i] = backend_type::zero();
        }
        return out;
    }

    [[nodiscard]] static basic_vec set1(T v) noexcept {
        basic_vec out;
        for (std::size_t i = 0; i < num_regs; ++i) {
            out.r_[i] = backend_type::set1(v);
        }
        return out;
    }

    /// Broadcast from a scalar pointer - the multi-register analogue of a
    /// broadcast load, and the reason a kernel can take either a value or an
    /// address without branching.
    [[nodiscard]] static basic_vec load1(const T* p) noexcept {
        return set1(*p);
    }

    /// Load N lanes.  Only N elements are read: when the last register is
    /// partial, the tail is staged through a zeroed buffer rather than read
    /// past the end, so a load of f32<6> from a 6-element array is safe.
    [[nodiscard]] static basic_vec load(const T* p) noexcept {
        basic_vec out;
        for (std::size_t i = 0; i < num_regs; ++i) {
            if (i < full_regs) {
                out.r_[i] = backend_type::load(p + i * reg_lanes);
            } else {
                alignas(64) T tmp[reg_lanes] = {};
                for (std::size_t j = 0; j < tail_lanes; ++j) {
                    tmp[j] = p[i * reg_lanes + j];
                }
                out.r_[i] = backend_type::load(tmp);
            }
        }
        return out;
    }

    /// Alias kept for readability at call sites that care about alignment.
    /// The backend is free to emit the aligned form when it can prove it.
    [[nodiscard]] static basic_vec loadu(const T* p) noexcept { return load(p); }
    [[nodiscard]] static basic_vec loada(const T* p) noexcept { return load(p); }

    /// Lane indices: 0, 1, 2, ...  Useful for float ramps (lerp parameters,
    /// distance fields) as much as for integer address arithmetic.
    [[nodiscard]] static basic_vec iota() noexcept {
        basic_vec out;
        for (std::size_t i = 0; i < num_regs; ++i) {
            auto r = backend_type::iota();
            if (i != 0) {
                r = backend_type::add(r, backend_type::set1(
                    static_cast<T>(i * reg_lanes)));
            }
            out.r_[i] = r;
        }
        return out;
    }

    /// A vector whose lane i is `first + i * step`.  For integers this is the
    /// address-arithmetic pattern that shows up in every gather-shaped loop.
    ///
    /// Constrained on the backend having a multiply, because that is what the
    /// ramp is built from: on AVX2 a 64-bit lane has no multiply, and this
    /// being a clear "constraints not satisfied" beats a confusing "no member
    /// named mul" from three template levels down.
    [[nodiscard]] static basic_vec iota(T first, T step) noexcept
        requires BackendHasMul<T, Abi> {
        basic_vec out;
        for (std::size_t i = 0; i < num_regs; ++i) {
            auto r = backend_type::iota();
            const T base = static_cast<T>(first + static_cast<T>(i * reg_lanes) * step);
            r = backend_type::add(backend_type::mul(r, backend_type::set1(step)),
                                  backend_type::set1(base));
            out.r_[i] = r;
        }
        return out;
    }

    // ---------------------------------------------------------------- memory
    /// Store N lanes, and no more: a partial final register writes only its
    /// live lanes.
    void store(T* p) const noexcept {
        for (std::size_t i = 0; i < num_regs; ++i) {
            if (i < full_regs) {
                backend_type::store(p + i * reg_lanes, r_[i]);
            } else {
                alignas(64) T tmp[reg_lanes];
                backend_type::store(tmp, r_[i]);
                for (std::size_t j = 0; j < tail_lanes; ++j) {
                    p[i * reg_lanes + j] = tmp[j];
                }
            }
        }
    }
    void storeu(T* p) const noexcept { store(p); }
    void storea(T* p) const noexcept { store(p); }

    [[nodiscard]] std::array<T, N> to_array() const noexcept {
        std::array<T, N> out{};
        for (std::size_t i = 0; i < num_regs; ++i) {
            const std::size_t live = (i < full_regs) ? reg_lanes : tail_lanes;
            alignas(64) T tmp[reg_lanes];
            backend_type::store(tmp, r_[i]);
            for (std::size_t j = 0; j < live; ++j) {
                out[i * reg_lanes + j] = tmp[j];
            }
        }
        return out;
    }

    /// Lane access.  Unchecked, like std::span: this is the slow path, and the
    /// kernels that matter never use it.
    [[nodiscard]] T operator[](std::size_t lane) const noexcept {
        const std::size_t ri = lane / reg_lanes;
        const std::size_t li = lane % reg_lanes;
        alignas(64) T tmp[reg_lanes];
        backend_type::store(tmp, r_[ri]);
        return tmp[li];
    }

    void set_lane(std::size_t lane, T v) noexcept {
        const std::size_t ri = lane / reg_lanes;
        const std::size_t li = lane % reg_lanes;
        alignas(64) T tmp[reg_lanes];
        backend_type::store(tmp, r_[ri]);
        tmp[li] = v;
        r_[ri] = backend_type::load(tmp);
    }

    /// The underlying registers, for the reduction helpers and for intrinsics
    /// interop.  A plain array reference rather than a std::array: see the note
    /// on the member declaration.
    [[nodiscard]] const reg_type (&raw() const noexcept)[num_regs] { return r_; }

    // -------------------------------------------------------------- fold core
    template <class F>
    [[nodiscard]] static basic_vec binop(const basic_vec& a, const basic_vec& b,
                                         F f) noexcept {
        basic_vec out;
        for (std::size_t i = 0; i < num_regs; ++i) {
            out.r_[i] = f(a.r_[i], b.r_[i]);
        }
        return out;
    }

    template <class F>
    [[nodiscard]] static basic_vec unop(const basic_vec& a, F f) noexcept {
        basic_vec out;
        for (std::size_t i = 0; i < num_regs; ++i) out.r_[i] = f(a.r_[i]);
        return out;
    }

    template <class F>
    [[nodiscard]] static mask_type cmpop(const basic_vec& a, const basic_vec& b,
                                         F f) noexcept {
        return mask_type::generate(
            [&](std::size_t i) { return f(a.r_[i], b.r_[i]); });
    }

    // ------------------------------------------------------------ arithmetic
    [[nodiscard]] friend basic_vec operator+(const basic_vec& a, const basic_vec& b) noexcept {
        return binop(a, b, [](reg_type x, reg_type y) { return backend_type::add(x, y); });
    }
    [[nodiscard]] friend basic_vec operator-(const basic_vec& a, const basic_vec& b) noexcept {
        return binop(a, b, [](reg_type x, reg_type y) { return backend_type::sub(x, y); });
    }
    [[nodiscard]] friend basic_vec operator*(const basic_vec& a, const basic_vec& b) noexcept {
        return binop(a, b, [](reg_type x, reg_type y) { return backend_type::mul(x, y); });
    }
    [[nodiscard]] friend basic_vec operator/(const basic_vec& a, const basic_vec& b) noexcept
        requires std::floating_point<T> {
        return binop(a, b, [](reg_type x, reg_type y) { return backend_type::div(x, y); });
    }

    [[nodiscard]] basic_vec operator-() const noexcept {
        return unop(*this, [](reg_type x) { return backend_type::neg(x); });
    }
    [[nodiscard]] basic_vec operator+() const noexcept { return *this; }

    basic_vec& operator+=(const basic_vec& o) noexcept { *this = *this + o; return *this; }
    basic_vec& operator-=(const basic_vec& o) noexcept { *this = *this - o; return *this; }
    basic_vec& operator*=(const basic_vec& o) noexcept { *this = *this * o; return *this; }
    basic_vec& operator/=(const basic_vec& o) noexcept
        requires std::floating_point<T> { *this = *this / o; return *this; }

    // ---------------------------------------------------------------- bitwise
    [[nodiscard]] friend basic_vec operator&(const basic_vec& a, const basic_vec& b) noexcept {
        return binop(a, b, [](reg_type x, reg_type y) { return backend_type::band(x, y); });
    }
    [[nodiscard]] friend basic_vec operator|(const basic_vec& a, const basic_vec& b) noexcept {
        return binop(a, b, [](reg_type x, reg_type y) { return backend_type::bor(x, y); });
    }
    [[nodiscard]] friend basic_vec operator^(const basic_vec& a, const basic_vec& b) noexcept {
        return binop(a, b, [](reg_type x, reg_type y) { return backend_type::bxor(x, y); });
    }
    /// `andnot(a, b)` == `(~a) & b`, the Intel operand order.
    [[nodiscard]] friend basic_vec andnot(const basic_vec& a, const basic_vec& b) noexcept {
        return binop(a, b, [](reg_type x, reg_type y) { return backend_type::bandnot(x, y); });
    }

    /// Shift by a uniform amount; lane-varying shifts are a gather-sized
    /// problem and are not this API.
    [[nodiscard]] basic_vec operator<<(int n) const noexcept
        requires std::integral<T> {
        return unop(*this, [n](reg_type x) { return backend_type::shl(x, n); });
    }
    [[nodiscard]] basic_vec operator>>(int n) const noexcept
        requires std::integral<T> {
        return unop(*this, [n](reg_type x) { return backend_type::shr_logical(x, n); });
    }
    [[nodiscard]] basic_vec shr_arith(int n) const noexcept
        requires std::signed_integral<T> {
        return unop(*this, [n](reg_type x) { return backend_type::shr_arith(x, n); });
    }

    basic_vec& operator&=(const basic_vec& o) noexcept { *this = *this & o; return *this; }
    basic_vec& operator|=(const basic_vec& o) noexcept { *this = *this | o; return *this; }
    basic_vec& operator^=(const basic_vec& o) noexcept { *this = *this ^ o; return *this; }
    basic_vec& operator<<=(int n) noexcept requires std::integral<T> { *this = *this << n; return *this; }
    basic_vec& operator>>=(int n) noexcept requires std::integral<T> { *this = *this >> n; return *this; }

    // ----------------------------------------------------------- comparisons
    [[nodiscard]] friend mask_type operator==(const basic_vec& a, const basic_vec& b) noexcept {
        return cmpop(a, b, [](reg_type x, reg_type y) { return backend_type::cmpeq(x, y); });
    }
    [[nodiscard]] friend mask_type operator!=(const basic_vec& a, const basic_vec& b) noexcept {
        return cmpop(a, b, [](reg_type x, reg_type y) { return backend_type::cmpne(x, y); });
    }
    [[nodiscard]] friend mask_type operator<(const basic_vec& a, const basic_vec& b) noexcept {
        return cmpop(a, b, [](reg_type x, reg_type y) { return backend_type::cmplt(x, y); });
    }
    [[nodiscard]] friend mask_type operator<=(const basic_vec& a, const basic_vec& b) noexcept {
        return cmpop(a, b, [](reg_type x, reg_type y) { return backend_type::cmple(x, y); });
    }
    [[nodiscard]] friend mask_type operator>(const basic_vec& a, const basic_vec& b) noexcept {
        return cmpop(a, b, [](reg_type x, reg_type y) { return backend_type::cmpgt(x, y); });
    }
    [[nodiscard]] friend mask_type operator>=(const basic_vec& a, const basic_vec& b) noexcept {
        return cmpop(a, b, [](reg_type x, reg_type y) { return backend_type::cmpge(x, y); });
    }

    // ------------------------------------------------------- mask selection
    [[nodiscard]] static basic_vec select(const mask_type& m, const basic_vec& a,
                                          const basic_vec& b) noexcept {
        basic_vec out;
        for (std::size_t i = 0; i < num_regs; ++i) {
            out.r_[i] = backend_type::select(m.raw()[i], a.r_[i], b.r_[i]);
        }
        return out;
    }

    // ------------------------------------------------- elements a kernel wants
    [[nodiscard]] basic_vec sqrt() const noexcept requires std::floating_point<T> {
        return unop(*this, [](reg_type x) { return backend_type::sqrt(x); });
    }
    /// Raw hardware estimate: ~12 significant bits on AVX2, ~14 on AVX-512, and
    /// exact on the scalar path.  Feed it to math::rsqrt for a usable result;
    /// on its own it is a seed, not an answer.
    [[nodiscard]] basic_vec rsqrt_approx() const noexcept requires std::floating_point<T> {
        return unop(*this, [](reg_type x) { return backend_type::rsqrt_approx(x); });
    }
    [[nodiscard]] basic_vec rcp_approx() const noexcept requires std::floating_point<T> {
        return unop(*this, [](reg_type x) { return backend_type::rcp_approx(x); });
    }
    [[nodiscard]] basic_vec abs() const noexcept {
        return unop(*this, [](reg_type x) { return backend_type::abs(x); });
    }
    [[nodiscard]] basic_vec min(const basic_vec& o) const noexcept {
        return binop(*this, o, [](reg_type x, reg_type y) { return backend_type::min(x, y); });
    }
    [[nodiscard]] basic_vec max(const basic_vec& o) const noexcept {
        return binop(*this, o, [](reg_type x, reg_type y) { return backend_type::max(x, y); });
    }
    [[nodiscard]] basic_vec fma(const basic_vec& b, const basic_vec& c) const noexcept
        requires std::floating_point<T> {
        basic_vec out;
        for (std::size_t i = 0; i < num_regs; ++i) {
            out.r_[i] = backend_type::fma(r_[i], b.r_[i], c.r_[i]);
        }
        return out;
    }

private:
    /// One hardware register per slot.
    ///
    /// A plain C array, NOT std::array, and the difference is not cosmetic:
    /// GCC emits -Wignored-attributes for every vector type used as a template
    /// argument, because a container cannot honour the may_alias attribute on
    /// __m256.  Wrapping it in a struct does not help - the wrapped type is
    /// still a template argument.  A raw array is the only spelling that keeps
    /// the warning out of the header, and a header-only library has no business
    /// dictating its consumers' warning flags.  The build enables
    /// -Wignored-attributes precisely so a regression here is caught.
    reg_type r_[num_regs];
};

// ---------------------------------------------------------------- free forms
// Kernels read better as `sqrt(v)` than `v.sqrt()` in expression context, and
// generic code that is agnostic about whether it holds a scalar or a vector
// needs the free form to exist.

template <SimdVec V>
[[nodiscard]] inline V sqrt(const V& v) noexcept requires std::floating_point<typename V::value_type> {
    return v.sqrt();
}
template <SimdVec V>
[[nodiscard]] inline V abs(const V& v) noexcept { return v.abs(); }
template <SimdVec V>
[[nodiscard]] inline V min(const V& a, const V& b) noexcept { return a.min(b); }
template <SimdVec V>
[[nodiscard]] inline V max(const V& a, const V& b) noexcept { return a.max(b); }
template <SimdVec V>
[[nodiscard]] inline V fma(const V& a, const V& b, const V& c) noexcept
    requires std::floating_point<typename V::value_type> {
    return a.fma(b, c);
}
template <SimdVec V>
[[nodiscard]] inline V select(const typename V::mask_type& m, const V& a, const V& b) noexcept {
    return V::select(m, a, b);
}

// -------------------------------------------------------------- named widths

/// Float vector of N lanes on the widest ABI this build may emit.
template <std::size_t N> using f32 = basic_vec<float, N>;
template <std::size_t N> using f64 = basic_vec<double, N>;
template <std::size_t N> using i32 = basic_vec<std::int32_t, N>;
template <std::size_t N> using i64 = basic_vec<std::int64_t, N>;
template <std::size_t N> using u32 = basic_vec<std::uint32_t, N>;
template <std::size_t N> using u64 = basic_vec<std::uint64_t, N>;

/// Lanes one register holds for T on the native ABI.
template <class T>
inline constexpr std::size_t native_width_v = backend<T, native_abi>::lanes;

/// The vector that exactly fills one native register - `std::simd`'s
/// `native_simd` under a different name.
template <class T>
using native_vec = basic_vec<T, native_width_v<T>>;

using native_f32 = native_vec<float>;
using native_f64 = native_vec<double>;
using native_i32 = native_vec<std::int32_t>;
using native_i64 = native_vec<std::int64_t>;

} // namespace vectis
