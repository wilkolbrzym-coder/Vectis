cmake_minimum_required(VERSION 3.24)

project(vectis
    VERSION 0.1.0
    DESCRIPTION "Compile-time SIMD engine for C++"
    LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(VECTIS_BUILD_TESTS      "Build the Vectis test suite"           ON)
option(VECTIS_BUILD_BENCHMARKS "Build the Vectis benchmarks"           ON)
option(VECTIS_BUILD_TOOLS      "Build the vectis-cpuinfo tool"         ON)
option(VECTIS_ENABLE_ASM       "Build hand-written assembly kernels"   ON)
option(VECTIS_WERROR           "Treat warnings as errors"              OFF)

# ---------------------------------------------------------------------------
# ISA tier
#
# One knob, because the two things it controls must never disagree: the -m
# flags the compiler is given, and the ceiling Vectis is allowed to emit.  Set
# them separately and you get a binary whose ISA level claims AVX-512 while the
# compiler never emitted a single ZMM instruction.
#
#   auto    - baseline for the build machine's floor (default: x86-64-v3)
#   native  - -march=native, the fastest this exact CPU can do
#   scalar  - no vector ISA at all; the reference path and the correctness oracle
#   avx2    - AVX2 + FMA, 256-bit
#   avx512  - AVX-512 F/BW/DQ/VL, 512-bit
# ---------------------------------------------------------------------------
# There is deliberately no sse42 tier.  A 128-bit backend would be a third
# parallel implementation of every operation to serve hardware that in practice
# always has AVX2 (Haswell, 2013), and the multi-register path is already
# exercised by wide vectors at the scalar and AVX2 tiers.  isa_level::sse42
# still exists so the runtime feature report can name a pre-AVX2 CPU honestly.
set(VECTIS_ISA_TIER "auto" CACHE STRING
    "ISA tier to build for: auto|native|scalar|avx2|avx512")
set_property(CACHE VECTIS_ISA_TIER PROPERTY STRINGS
    auto native scalar avx2 avx512)

set(VECTIS_BASELINE_ISA "x86-64-v3" CACHE STRING
    "Floor used by the 'auto' tier (x86-64-v2, x86-64-v3, x86-64-v4)")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# Compiler checks
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(VECTIS_IS_GNU TRUE)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(VECTIS_IS_CLANG TRUE)
else()
    message(FATAL_ERROR
        "Vectis targets GCC and Clang (found ${CMAKE_CXX_COMPILER_ID})")
endif()

set(VECTIS_ON_X86 FALSE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|i[3-6]86)$")
    set(VECTIS_ON_X86 TRUE)
endif()

# The AVX-512 quartet.  Anything narrower is not a tier we support: F alone
# cannot express byte-granular blends or efficient 32-bit lane compares.
set(VECTIS_AVX512_FLAGS
    -mavx512f -mavx512bw -mavx512dq -mavx512vl)
set(VECTIS_AVX512_OFF_FLAGS
    -mno-avx512f -mno-avx512bw -mno-avx512dq -mno-avx512vl)
set(VECTIS_AVX_OFF_FLAGS -mno-avx -mno-avx2 -mno-fma)

# ---------------------------------------------------------------------------
# Resolve tier -> flags + ceiling
# ---------------------------------------------------------------------------
set(VECTIS_ARCH_FLAGS "")
set(VECTIS_MAX_ISA_LEVEL 99)

if(NOT VECTIS_ON_X86)
    # Non-x86: the scalar backend is the whole library.  Everything above it is
    # architecture-specific and simply is not emitted.
    set(VECTIS_MAX_ISA_LEVEL 0)
    set(VECTIS_ISA_TIER_RESOLVED "scalar (non-x86 host)")
elseif(VECTIS_ISA_TIER STREQUAL "native")
    list(APPEND VECTIS_ARCH_FLAGS -march=native)
    set(VECTIS_ISA_TIER_RESOLVED "native")
elseif(VECTIS_ISA_TIER STREQUAL "scalar")
    list(APPEND VECTIS_ARCH_FLAGS ${VECTIS_AVX_OFF_FLAGS} ${VECTIS_AVX512_OFF_FLAGS} -mno-sse4.2)
    set(VECTIS_MAX_ISA_LEVEL 0)
    set(VECTIS_ISA_TIER_RESOLVED "scalar")
elseif(VECTIS_ISA_TIER STREQUAL "avx2")
    list(APPEND VECTIS_ARCH_FLAGS -march=x86-64-v3 ${VECTIS_AVX512_OFF_FLAGS})
    set(VECTIS_MAX_ISA_LEVEL 2)
    set(VECTIS_ISA_TIER_RESOLVED "avx2")
elseif(VECTIS_ISA_TIER STREQUAL "avx512")
    list(APPEND VECTIS_ARCH_FLAGS -march=x86-64-v4)
    set(VECTIS_MAX_ISA_LEVEL 3)
    set(VECTIS_ISA_TIER_RESOLVED "avx512")
else() # auto
    list(APPEND VECTIS_ARCH_FLAGS -march=${VECTIS_BASELINE_ISA})
    set(VECTIS_ISA_TIER_RESOLVED "auto (${VECTIS_BASELINE_ISA})")
endif()

set(VECTIS_WARN_FLAGS
    -Wall -Wextra -Wpedantic
    -Wshadow -Wconversion -Wsign-conversion
    -Wcast-align -Wunused -Wdouble-promotion
    # Enabled deliberately.  The register storage in basic_vec/basic_mask is a
    # plain C array rather than a std::array precisely so that no vector type
    # ever appears as a template argument, and this warning is what proves it -
    # a header-only library must not force -Wno-... on its consumers.
    -Wignored-attributes)
if(VECTIS_WERROR)
    list(APPEND VECTIS_WARN_FLAGS -Werror)
endif()

# ---------------------------------------------------------------------------
# Header-only library
# ---------------------------------------------------------------------------
add_library(vectis INTERFACE)
add_library(vectis::vectis ALIAS vectis)

target_include_directories(vectis INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# The language floor is C++20, and that is tested (see the `standards` CI job).
# C++26 is what this project is *built* at, not what it *requires*: the headers
# work at C++20, use C++23 features where the compiler offers them, and will use
# C++26 reflection the day it ships.
target_compile_features(vectis INTERFACE cxx_std_20)

# Arch and warning flags are BUILD_INTERFACE only.  They are how this project
# builds itself; a consumer who installs the package has their own opinions
# about -march and -Wconversion, and a library that forces them on every
# downstream target is a library people work around.
target_compile_options(vectis INTERFACE
    $<BUILD_INTERFACE:${VECTIS_ARCH_FLAGS}>
    $<BUILD_INTERFACE:${VECTIS_WARN_FLAGS}>)

# NOTE: VECTIS_MAX_ISA_LEVEL is deliberately NOT defined here.  The -m flags
# already are the cap - `-mno-avx512f` makes __AVX512F__ undefined, which is
# what the library's feature detection reads.  Setting a second, macro-level cap
# would create two sources of truth for one decision, and they would eventually
# disagree.  The macro still exists in config.hpp for a user who needs to clamp
# without touching compiler flags.

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++26" VECTIS_HAS_CXX26)
if(VECTIS_HAS_CXX26)
    target_compile_options(vectis INTERFACE $<BUILD_INTERFACE:-std=c++26>)
endif()

# ---------------------------------------------------------------------------
# Hand-written assembly kernel layer
#
# Deliberately a separate target: the .S files are assembled with their own
# flags and linked in only when the tier can use them, and - crucially - they
# can be dropped from the build entirely to A/B the same kernel against its
# intrinsics version.
# ---------------------------------------------------------------------------
if(VECTIS_ENABLE_ASM AND VECTIS_ON_X86 AND VECTIS_MAX_ISA_LEVEL GREATER_EQUAL 2)
    enable_language(ASM)
    add_library(vectis_asm OBJECT src/asm/avx2_kernels.S)
    target_include_directories(vectis_asm PRIVATE include)
    target_compile_options(vectis_asm PRIVATE -mavx2 -mfma)
    target_compile_definitions(vectis_asm PRIVATE
        VECTIS_MAX_ISA_LEVEL=${VECTIS_MAX_ISA_LEVEL})
    set(VECTIS_ASM_AVAILABLE TRUE)
else()
    set(VECTIS_ASM_AVAILABLE FALSE)
endif()

if(VECTIS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(VECTIS_BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()

if(VECTIS_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

# ---------------------------------------------------------------------------
# Install
#
# A header-only package that cannot be found with find_package() is only half a
# library.  CI installs it and builds a separate project against it, because
# "it compiles here" and "it compiles for someone who installed it" are
# different claims.
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(DIRECTORY include/vectis
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp")

install(TARGETS vectis EXPORT vectisTargets)

install(EXPORT vectisTargets
    FILE vectisTargets.cmake
    NAMESPACE vectis::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vectis)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/vectisConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/vectisConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vectis)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/vectisConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/vectisConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/vectisConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vectis)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "================= Vectis ${PROJECT_VERSION} =================")
message(STATUS "  compiler      : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  host          : ${CMAKE_SYSTEM_PROCESSOR} (x86=${VECTIS_ON_X86})")
message(STATUS "  C++26         : ${VECTIS_HAS_CXX26}")
message(STATUS "  ISA tier      : ${VECTIS_ISA_TIER_RESOLVED}")
message(STATUS "  ISA ceiling   : ${VECTIS_MAX_ISA_LEVEL} (0=scalar 1=sse42 2=avx2 3=avx512, 99=uncapped)")
message(STATUS "  arch flags    : ${VECTIS_ARCH_FLAGS}")
message(STATUS "  asm kernels   : ${VECTIS_ASM_AVAILABLE}")
message(STATUS "  build type    : ${CMAKE_BUILD_TYPE}")
message(STATUS "  tests / bench : ${VECTIS_BUILD_TESTS} / ${VECTIS_BUILD_BENCHMARKS}")
message(STATUS "====================================================")
message(STATUS "")
