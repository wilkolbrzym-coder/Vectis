# ===========================================================================
# Vectis tests
# ===========================================================================

add_executable(vectis_tests
    test_main.cpp
    test_simd_core.cpp
    test_simd_avx2.cpp
    test_simd_avx512.cpp
    test_soa.cpp
    test_soa_reflect.cpp
    test_asm_parity.cpp)

# The AVX-512 battery gets its own translation unit with AVX-512 enabled, so one
# test binary covers every backend on any host - see test_simd_avx512.cpp for
# why a target attribute is not enough.  Tolerated to be absent: on a compiler
# that does not accept these flags the file compiles itself out.
include(CheckCXXCompilerFlag)

# The backend batteries are x86-only, and so are their flags.  On aarch64 these
# -m options are *unused command-line arguments*, which clang reports as an
# error under -Werror - so the guard is not tidiness, it is what makes the
# macOS and ARM jobs build at all.  The sources compile to nothing there
# anyway; see the VECTIS_USE_AVX2 guards inside them.
if(VECTIS_ON_X86)
    check_cxx_compiler_flag("-mavx2" VECTIS_CC_HAS_AVX2)
    if(VECTIS_CC_HAS_AVX2)
        # The AVX2 battery wants to be a *pure* 256-bit unit, which means
        # disabling AVX-512 in it.  That is only safe to do when the tier does
        # not already have AVX-512 turned on: on a machine whose -march=native
        # reports AVX10.1-512, GCC 14 rejects -mno-avx512f outright with
        # "-mno-evex512 cannot disable AVX10 instructions when AVX10.1-512 is
        # available", and with -Werror that is a red build.  So the disable
        # flags are added only for the explicit avx2 tier, where they cannot
        # conflict; elsewhere the unit keeps its baseline, and nothing breaks
        # because the backend uses explicit _mm256_* intrinsics regardless.
        if(VECTIS_ISA_TIER STREQUAL "avx2")
            set_source_files_properties(test_simd_avx2.cpp PROPERTIES
                COMPILE_OPTIONS "-mavx2;-mfma;-mno-avx512f;-mno-avx512bw;-mno-avx512dq;-mno-avx512vl")
        else()
            set_source_files_properties(test_simd_avx2.cpp PROPERTIES
                COMPILE_OPTIONS "-mavx2;-mfma")
        endif()
    endif()
endif()

if(VECTIS_ON_X86)
    check_cxx_compiler_flag("-mavx512f" VECTIS_CC_HAS_AVX512)
    if(VECTIS_CC_HAS_AVX512)
        set_source_files_properties(test_simd_avx512.cpp PROPERTIES
            COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mavx512vl")
    else()
        message(STATUS "tests: compiler has no AVX-512 support, that battery is out")
    endif()
endif()

target_link_libraries(vectis_tests PRIVATE vectis::vectis)

# The assembly kernels are linked in when the build has them, so the parity
# tests can compare asm against intrinsics in the same binary.  When they are
# off, the tests that reference them compile themselves out.
if(VECTIS_ASM_AVAILABLE)
    target_sources(vectis_tests PRIVATE $<TARGET_OBJECTS:vectis_asm>)
    target_compile_definitions(vectis_tests PRIVATE VECTIS_HAS_ASM_KERNELS=1)
endif()

# Whole suite.
add_test(NAME vectis_tests COMMAND vectis_tests)

# Focused entry points, so a CI job or a developer can isolate one concern.
# The names matter: .github/workflows/ci.yml runs `ctest -R asm`.
add_test(NAME vectis_canaries COMMAND vectis_tests --filter canary)
add_test(NAME vectis_math     COMMAND vectis_tests --filter math)
add_test(NAME vectis_avx512   COMMAND vectis_tests --filter avx512)
add_test(NAME vectis_soa    COMMAND vectis_tests --filter soa)
add_test(NAME vectis_asm_parity COMMAND vectis_tests --filter asm)

set_tests_properties(
    vectis_canaries vectis_math vectis_avx512 vectis_soa vectis_asm_parity
    PROPERTIES LABELS "focused")
