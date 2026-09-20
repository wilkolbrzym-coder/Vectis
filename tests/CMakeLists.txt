# ===========================================================================
# Vectis tests
# ===========================================================================

add_executable(vectis_tests
    test_main.cpp
    test_simd_core.cpp
    test_simd_avx2.cpp
    test_simd_avx512.cpp
    test_soa.cpp
    test_asm_parity.cpp)

# The AVX-512 battery gets its own translation unit with AVX-512 enabled, so one
# test binary covers every backend on any host - see test_simd_avx512.cpp for
# why a target attribute is not enough.  Tolerated to be absent: on a compiler
# that does not accept these flags the file compiles itself out.
include(CheckCXXCompilerFlag)

# AVX2 battery: its own translation unit, AVX-512 explicitly off so it is a pure
# 256-bit unit whatever the library tier is.
check_cxx_compiler_flag("-mavx2" VECTIS_CC_HAS_AVX2)
if(VECTIS_CC_HAS_AVX2)
    set_source_files_properties(test_simd_avx2.cpp PROPERTIES
        COMPILE_OPTIONS "-mavx2;-mfma;-mno-avx512f;-mno-avx512bw;-mno-avx512dq;-mno-avx512vl")
endif()

check_cxx_compiler_flag("-mavx512f" VECTIS_CC_HAS_AVX512)
if(VECTIS_CC_HAS_AVX512)
    set_source_files_properties(test_simd_avx512.cpp PROPERTIES
        COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mavx512vl")
else()
    message(STATUS "tests: compiler has no AVX-512 support, that battery is out")
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
