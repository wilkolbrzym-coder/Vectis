// ===========================================================================
// Vectis - compile-time SIMD engine
// simd/reduce.hpp - collapsing a vector to one scalar
// ===========================================================================
//
// Reductions are where the padding lanes of a partially-filled final register
// would silently corrupt the answer.  Every function here therefore reduces
// exactly N lanes: the partial register - if there is one - has only its live
// lanes read.  Whole registers are folded with vector operations, and where a
// fold order would change the answer - the min and max reductions, whose rule is
// not associative - the data decides which of two orders is used.  See
// reduce_ordered.
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

/// True when any lane of any full register is a NaN or a zero.
///
/// This runs on the fast path of reduce_min/reduce_max, so it has to be cheaper
/// than the fold it guards, and it is written at the register level for exactly
/// that reason: comparing whole `basic_vec`s materialises a mask object per
/// operand and ends up costing more than the fold it was supposed to protect.
/// Two register comparisons and two mask combine steps per register, with one
/// mask_bits at the end, is the whole check - and it is what decides the branch,
/// up front, so that the two orders stay separate functions.  Merging them into
/// one body, or deciding afterwards and re-running, both cost the fast path
/// several times its speed.
///
/// `cmpeq(r, zero)` is true for both signs of zero, which is what makes it the
/// right test: the sign is the thing that makes a zero order-sensitive.
template <class B, class V>
[[nodiscard]] inline bool any_nan_or_zero(const V& v) noexcept {
    const auto z = B::zero();
    auto bad = B::mask_false();
    for (std::size_t i = 0; i < V::full_regs; ++i) {
        const auto r = v.raw()[i];
        bad = B::mask_or(bad, B::cmpne(r, r));   // true only for NaN
        bad = B::mask_or(bad, B::cmpeq(r, z));   // true for +0.0 and -0.0
    }
    return B::mask_bits(bad) != 0;
}

/// Every lane in turn, register after register: the order the oracle folds in.
///
/// Separate from the fold below rather than a branch inside it, so that each
/// order compiles on its own.  Sharing one body between them cost the fast path
/// five times its speed for a branch that almost always went the other way.
template <SimdVec V, class Better>
[[nodiscard]] inline typename V::value_type
reduce_in_lane_order(const V& v, Better better) noexcept {
    using B = typename V::backend_type;
    using T = typename V::value_type;

    T best{};
    bool seeded = false;
    auto consider = [&](T x) {
        if (!seeded) { best = x; seeded = true; }
        else if (better(x, best)) { best = x; }
    };

    if constexpr (V::full_regs > 0) {
        alignas(64) T tmp[B::lanes];
        for (std::size_t i = 0; i < V::full_regs; ++i) {
            B::store(tmp, v.raw()[i]);
            for (std::size_t j = 0; j < B::lanes; ++j) consider(tmp[j]);
        }
    }
    // The partial register contributes only its live lanes.
    if constexpr (V::tail_lanes > 0) {
        alignas(64) T tail[B::lanes];
        read_tail(v, tail);
        for (std::size_t j = 0; j < V::tail_lanes; ++j) consider(tail[j]);
    }
    return best;
}

/// Fold the whole registers lane-wise, scan the accumulator once, then the tail.
///
/// Only ever called when the guard in reduce_ordered has established that this
/// order gives the same answer as the lane-by-lane one.
template <SimdVec V, class VecFold, class Better>
[[nodiscard]] inline typename V::value_type
reduce_folded(const V& v, VecFold vec_fold, Better better) noexcept {
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
    // The partial register is scanned in lane order, which is the oracle's order
    // for it either way, so it needs no guard of its own.
    if constexpr (V::tail_lanes > 0) {
        alignas(64) T tail[B::lanes];
        read_tail(v, tail);
        for (std::size_t j = 0; j < V::tail_lanes; ++j) consider(tail[j]);
    }
    return best;
}

/// Shared shape for min and max.
///
/// There are two orders the lanes can be folded in, and only one of them is
/// right for every input.
///
/// The rule is `(a < b) ? a : b` - keep the right operand unless the left one
/// strictly beats it - and it is neither associative nor commutative.  With a
/// NaN or a signed zero in the data the answer depends on where the operands
/// sit, so the fold order is part of the result and not an implementation
/// detail.  The scalar backend folds in lane order and it is the oracle:
/// reduce_min and reduce_max have to return its answer on every tier, which is
/// the whole point of the battery they are tested by.
///
/// Folding the registers lane-wise first is faster and answers identically for
/// every input where the two orders cannot disagree: NaN-free data whose lanes
/// include no zero.  The minimum there is unique, and equal values have equal bit
/// patterns, so the fold has nothing left to be order-sensitive about.  Folding
/// without those conditions is the bug that made a 16-lane AVX2 vector holding
/// `{100,50,60,0.5,70,80,90,95,NaN,10,20,30,40,45,55,65}` reduce to `0.5` on
/// AVX2 and to `10` on scalar: the fold moves every lane past an earlier
/// column's NaN, and the sequential scan does not.
///
/// So the guard below is a correctness condition, not a hint - lane order when
/// the data can expose the difference, the fold when it cannot.  Integers never
/// take the lane-order path, because an integer order is total and the fold is
/// exact for them whatever the data; they pay nothing for it, and neither does a
/// vector that fits in one register.
///
/// `better` is the scalar predicate - (candidate beats current) - and it has to
/// reproduce `(a < b) ? a : b` exactly, not approximate it with `candidate <
/// current`.  The tempting `candidate < current` keeps the left operand instead,
/// which disagrees on `-0.0` vs `+0.0` and on every NaN.
template <SimdVec V, class VecFold, class Better>
[[nodiscard]] inline typename V::value_type
reduce_ordered(const V& v, VecFold vec_fold, Better better) noexcept {
    using B = typename V::backend_type;
    using T = typename V::value_type;

    if constexpr (V::full_regs > 1 && std::floating_point<T>) {
        if (any_nan_or_zero<B>(v)) return reduce_in_lane_order(v, better);
    }
    return reduce_folded(v, vec_fold, better);
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
