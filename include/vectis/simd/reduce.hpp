// ===========================================================================
// Vectis - compile-time SIMD engine
// simd/reduce.hpp - collapsing a vector to one scalar
// ===========================================================================
//
// Reductions are where the padding lanes of a partially-filled final register
// would silently corrupt the answer.  Every function here therefore reduces
// exactly N lanes: whole registers are folded with vector operations, and the
// partial register - if there is one - has only its live lanes read.
//
// On the shape of the code: the horizontal step stages the register through a
// 64-byte aligned buffer and sums it scalar-wise.  A shuffle tree would avoid
// the store-forwarding stall, but it needs a different shuffle per lane width
// and per element type, and for the throughput-bound loops this library targets
// the compiler usually hides the store anyway.  It is measured in the
// benchmarks rather than assumed - see bench/bench_reduce.cpp.
//
// ===========================================================================
#pragma once

#include "math.hpp"
#include "vec.hpp"

namespace vectis {

namespace detail {

/// Read the live lanes of a partially-filled register.
template <class V>
inline void read_tail(const V& v, typename V::value_type* out) noexcept {
    using B = typename V::backend_type;
    using T = typename V::value_type;
    if constexpr (V::tail_lanes > 0) {
        alignas(64) T tmp[B::lanes];
        B::store(tmp, v.raw()[V::full_regs]);
        for (std::size_t j = 0; j < V::tail_lanes; ++j) out[j] = tmp[j];
    } else {
        (void)v;
        (void)out;
    }
}

} // namespace detail

/// Sum of all N lanes.
template <SimdVec V>
[[nodiscard]] inline typename V::value_type reduce_add(const V& v) noexcept {
    using B = typename V::backend_type;
    using T = typename V::value_type;

    T total = T{0};

    // Only the whole registers take part in the vector fold; the partial one is
    // read lane-wise so its padding never contributes.
    if constexpr (V::full_regs > 0) {
        auto acc = v.raw()[0];
        for (std::size_t i = 1; i < V::full_regs; ++i) {
            acc = B::add(acc, v.raw()[i]);
        }
        alignas(64) T tmp[B::lanes];
        B::store(tmp, acc);
        for (std::size_t j = 0; j < B::lanes; ++j) total += tmp[j];
    }
    if constexpr (V::tail_lanes > 0) {
        alignas(64) T tail[B::lanes];
        detail::read_tail(v, tail);
        for (std::size_t j = 0; j < V::tail_lanes; ++j) total += tail[j];
    }
    return total;
}

namespace detail {

/// Shared shape for min and max.
///
/// Two different folds are at work and they must not be confused: the register
/// fold combines *vectors* with the backend's min/max, while the horizontal step
/// combines *scalars*.  `better` is the scalar predicate - (candidate beats
/// current) - and it has to reproduce the hardware rule `(a < b) ? a : b`
/// exactly, not approximate it with `candidate < current`.
///
/// The difference is which operand wins a tie, and it is observable without any
/// NaN: scanning `[a, b]` left to right, the rule is `(a < b) ? a : b`, so `b`
/// survives unless `a` strictly beats it - i.e. the predicate is `!(current <
/// candidate)`.  The tempting `candidate < current` keeps `a` instead, which
/// disagrees with the register fold on `-0.0` vs `+0.0` and on every NaN, and
/// therefore makes reduce_min/reduce_max return different answers at different
/// tiers for the same logical input.  The scalar backend is the oracle; this
/// predicate is what keeps the horizontal step faithful to it.
template <SimdVec V, class VecFold, class Better>
[[nodiscard]] inline typename V::value_type
reduce_ordered(const V& v, VecFold vec_fold, Better better) noexcept {
    using B = typename V::backend_type;
    using T = typename V::value_type;

    T best{};
    bool seeded = false;
    auto consider = [&](T x) {
        if (!seeded) { best = x; seeded = true; }
        else if (better(x, best)) { best = x; }
    };

    if constexpr (V::full_regs > 0) {
        auto acc = v.raw()[0];
        for (std::size_t i = 1; i < V::full_regs; ++i) {
            acc = vec_fold(acc, v.raw()[i]);
        }
        alignas(64) T tmp[B::lanes];
        B::store(tmp, acc);
        for (std::size_t j = 0; j < B::lanes; ++j) consider(tmp[j]);
    }
    // The partial register contributes only its live lanes.
    if constexpr (V::tail_lanes > 0) {
        alignas(64) T tail[B::lanes];
        read_tail(v, tail);
        for (std::size_t j = 0; j < V::tail_lanes; ++j) consider(tail[j]);
    }
    return best;
}

} // namespace detail

template <SimdVec V>
[[nodiscard]] inline typename V::value_type reduce_min(const V& v) noexcept {
    using B = typename V::backend_type;
    return detail::reduce_ordered(
        v, [](auto a, auto b) { return B::min(a, b); },
        [](auto candidate, auto current) { return !(current < candidate); });
}

template <SimdVec V>
[[nodiscard]] inline typename V::value_type reduce_max(const V& v) noexcept {
    using B = typename V::backend_type;
    return detail::reduce_ordered(
        v, [](auto a, auto b) { return B::max(a, b); },
        [](auto candidate, auto current) { return !(current > candidate); });
}

/// True when every lane satisfies the mask - the "did all of these pass" test
/// that culling and validation code is built from.
template <SimdVec V>
[[nodiscard]] inline bool all_of(const typename V::mask_type& m) noexcept {
    return m.all();
}
template <SimdVec V>
[[nodiscard]] inline bool any_of(const typename V::mask_type& m) noexcept {
    return m.any();
}

// ------------------------------------------------------------ vector algebra
// The handful of operations that turn a lane-wise engine into something a
// geometry kernel can be written against.

/// Lane-wise dot product of two vectors of the same shape.
template <SimdVec V>
[[nodiscard]] inline typename V::value_type dot(const V& a, const V& b) noexcept
    requires std::floating_point<typename V::value_type> {
    return reduce_add(a * b);
}

template <SimdVec V>
[[nodiscard]] inline typename V::value_type length_sq(const V& v) noexcept
    requires std::floating_point<typename V::value_type> {
    return dot(v, v);
}

template <SimdVec V>
[[nodiscard]] inline typename V::value_type length(const V& v) noexcept
    requires std::floating_point<typename V::value_type> {
    return std::sqrt(length_sq(v));
}

/// Unit vector.  Uses the refined reciprocal square root, so it costs one
/// estimate plus a Newton step rather than a sqrt and a divide.
template <SimdVec V>
[[nodiscard]] inline V normalize(const V& v) noexcept
    requires std::floating_point<typename V::value_type> {
    return v * math::rsqrt(V::set1(length_sq(v)));
}

/// Fill N lanes from a buffer plus a scalar tail, the shape almost every real
/// loop has.  The tail is processed one lane at a time, which is what keeps the
/// vector path honest: no masked loads, no reads past the end.
///
/// `f` is called as f(vec_chunk_ptr) for each full vector and f(ptr) per
/// remaining element, so the caller stays in control of what the kernel does.
template <class T, std::size_t N, class VecFn, class TailFn>
inline void for_each_chunk(const T* data, std::size_t count, VecFn vec_fn,
                           TailFn tail_fn) noexcept {
    using V = basic_vec<T, N>;
    std::size_t i = 0;
    for (; i + N <= count; i += N) {
        vec_fn(V::load(data + i));
    }
    for (; i < count; ++i) tail_fn(data[i]);
}

/// Same, but the functor receives the vector and its lane offset, which is what
/// a kernel needs when it has to write to a second array.
template <class T, std::size_t N, class VecFn, class TailFn>
inline void for_each_chunk_indexed(const T* data, std::size_t count,
                                   VecFn vec_fn, TailFn tail_fn) noexcept {
    using V = basic_vec<T, N>;
    std::size_t i = 0;
    for (; i + N <= count; i += N) {
        vec_fn(V::load(data + i), i);
    }
    for (; i < count; ++i) tail_fn(data[i], i);
}

} // namespace vectis
