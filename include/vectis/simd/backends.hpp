// ===========================================================================
// Vectis - compile-time SIMD engine
// simd/backends.hpp - one include point for every backend that is available
// ===========================================================================
//
// Each backend header guards itself on VECTIS_USE_*, so including them all is
// free: on an AVX2 build, backend_avx512.hpp expands to nothing.  The scalar
// backend is always present, because it is the correctness oracle for every
// comparison test regardless of what the build tier is.
// ===========================================================================
#pragma once

#include "backend.hpp"
#include "backend_scalar.hpp"
#include "backend_avx2.hpp"
#include "backend_avx512.hpp"
