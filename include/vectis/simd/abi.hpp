// ===========================================================================
// Vectis - compile-time SIMD engine
// simd/abi.hpp - register-file strategy tags
// ===========================================================================
#pragma once

#include "../core/config.hpp"
#include "../core/concepts.hpp"

namespace vectis {

/// One lane per register: the reference implementation.
struct scalar_abi {
    static constexpr isa_level isa = isa_level::scalar;
    static constexpr const char* name = "scalar";
};

/// 128-bit registers (XMM).
struct sse42_abi {
    static constexpr isa_level isa = isa_level::sse42;
    static constexpr const char* name = "sse4.2";
};

/// 256-bit registers (YMM), with FMA.
struct avx2_abi {
    static constexpr isa_level isa = isa_level::avx2;
    static constexpr const char* name = "avx2";
};

/// 512-bit registers (ZMM), with mask registers.
struct avx512_abi {
    static constexpr isa_level isa = isa_level::avx512;
    static constexpr const char* name = "avx512";
};

/// The widest ABI this translation unit is allowed to emit.
///
/// Note the absence of an sse42_abi branch: there is no 128-bit backend, and
/// there will not be one (see the tier discussion in CMakeLists.txt).  A build
/// that can only reach SSE4.2 gets the scalar path, which is correct but not
/// wide.  sse42_abi exists as a name so the runtime feature report can describe
/// a pre-AVX2 host honestly, not as a code generation target.
#if defined(VECTIS_USE_AVX512)
using native_abi = avx512_abi;
#elif defined(VECTIS_USE_AVX2)
using native_abi = avx2_abi;
#else
using native_abi = scalar_abi;
#endif

/// Widest ABI available at all, for tests that want to name every tier.
template <isa_level L>
struct abi_of;
template <> struct abi_of<isa_level::scalar> { using type = scalar_abi; };
template <> struct abi_of<isa_level::sse42>  { using type = sse42_abi;  };
template <> struct abi_of<isa_level::avx2>   { using type = avx2_abi;   };
template <> struct abi_of<isa_level::avx512> { using type = avx512_abi; };

template <isa_level L>
using abi_of_t = typename abi_of<L>::type;

} // namespace vectis
