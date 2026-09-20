# ===========================================================================
# Vectis benchmarks
#
# These are not part of `ctest`: a benchmark on a shared CI runner reports the
# neighbours' noise as much as the kernel.  They are built so a human can run
# them on a quiet machine, and so CI can at least prove they still compile.
# ===========================================================================

add_library(vectis_bench_harness INTERFACE)
target_include_directories(vectis_bench_harness INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(bench_asm bench_asm.cpp)
target_link_libraries(bench_asm PRIVATE vectis::vectis vectis_bench_harness)

if(VECTIS_ASM_AVAILABLE)
    target_sources(bench_asm PRIVATE $<TARGET_OBJECTS:vectis_asm>)
    target_compile_definitions(bench_asm PRIVATE VECTIS_HAS_ASM_KERNELS=1)
endif()

add_executable(bench_simd bench_simd.cpp)
target_link_libraries(bench_simd PRIVATE vectis::vectis vectis_bench_harness)

add_executable(bench_soa bench_soa.cpp)
target_link_libraries(bench_soa PRIVATE vectis::vectis vectis_bench_harness)

# Build-only checks: benchmarks must not bit-rot just because nobody ran them.
add_test(NAME bench_asm_builds COMMAND bench_asm --size 1024)
add_test(NAME bench_simd_builds COMMAND bench_simd --quick)
add_test(NAME bench_soa_builds COMMAND bench_soa --quick)
set_tests_properties(bench_asm_builds bench_simd_builds bench_soa_builds
    PROPERTIES LABELS "bench" TIMEOUT 300)
