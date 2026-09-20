// ===========================================================================
// Vectis - is hand-written assembly worth it?
// ===========================================================================
//
// Three kernels, three spellings each, four working-set sizes chosen to land in
// L1, L2, L3 and DRAM.  The point is not to prove assembly is fast; it is to
// find out where it is, and to have a number rather than an opinion.
//
// The expected shape, stated in advance so the result can contradict it:
//
//   small sizes   compute-bound   - the add/mul/FMA ports are the limit, and
//                                   all three spellings should be within noise
//   large sizes   bandwidth-bound - the load/store units and DRAM are the
//                                   limit, and the speedup collapses to ~1.0
//                                   no matter how clever the instruction
//                                   selection is
//
// If a large-size run shows a big speedup, something is wrong with the
// measurement, not with the kernel.
//
// ===========================================================================
#include "bench_harness.hpp"

#include <vectis/core/cpu.hpp>
#include <vectis/kernels/asm_kernels.hpp>

#include <cstdio>
#include <cstring>
#include <vector>

using namespace vectis;

namespace {

/// Deterministic fill: a benchmark that cannot be re-run identically is not
/// a benchmark.
void fill(std::vector<float>& v, std::uint32_t seed) {
    std::uint32_t s = seed;
    for (auto& x : v) {
        s = s * 1664525u + 1013904223u;
        x = static_cast<float>(static_cast<double>(s >> 8) / 8388608.0 - 1.0);
    }
}

std::size_t parse_size(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--size") == 0) {
            return static_cast<std::size_t>(std::strtoull(argv[i + 1], nullptr, 10));
        }
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    std::printf("\nvectis benchmark: asm vs intrinsics vs scalar\n");
    std::printf("=============================================\n");
    {
        const auto b = cpu::brand();
        std::printf("  cpu      : %.*s\n", static_cast<int>(b.size()), b.data());
    }
    {
        const auto d = cpu::describe();
        std::printf("  features : %.*s\n", static_cast<int>(d.size()), d.data());
    }
    std::printf("  tier     : %s (compiled) / %s (runtime best)\n",
                to_string(compiled_isa), to_string(cpu::get().best));

    if (!kernels::runnable_here()) {
        std::printf("\n  SKIP: these kernels need AVX2 and this CPU has none.\n");
        return 0;
    }
    if (!kernels::asm_kernels_available()) {
        std::printf("\n  SKIP: built with VECTIS_ENABLE_ASM=OFF, so there is\n"
                    "        nothing to compare against.\n");
        return 0;
    }

    const std::size_t forced = parse_size(argc, argv);
    const std::vector<std::size_t> sizes =
        forced != 0 ? std::vector<std::size_t>{forced}
                    : std::vector<std::size_t>{1024, 32768, 262144, 4194304};

    for (const std::size_t n : sizes) {
        std::vector<float> a(n), b(n), y(n);
        fill(a, 12345u);
        fill(b, 67890u);
        fill(y, 24680u);

        const double kib = static_cast<double>(n) * 4.0 / 1024.0;
        char title[128];
        std::snprintf(title, sizeof title,
                      "dot_f32: n=%zu floats (%.1f KiB per array, %.0f KiB moved)",
                      n, kib, kib * 2.0);

        vbench::options opt;
        opt.elems = n;
        opt.bytes = n * 4 * 2;      // two arrays read

        auto make_dot = [&](const char* which) {
            return [&, which]() -> double {
                if (std::strcmp(which, "scalar") == 0) {
                    return static_cast<double>(kernels::dot_f32_scalar(a.data(), b.data(), n));
                }
                if (std::strcmp(which, "intrinsics") == 0) {
                    return static_cast<double>(kernels::dot_f32_intrinsics(a.data(), b.data(), n));
                }
                return static_cast<double>(vectis_asm_dot_f32(a.data(), b.data(), n));
            };
        };
        vbench::compare(title, opt, {"scalar", "intrinsics", "asm"}, make_dot);

        opt.bytes = n * 4 * 3;      // two reads + one write
        char title2[128];
        std::snprintf(title2, sizeof title2,
                      "saxpy_f32: n=%zu floats (%.1f KiB per array)", n, kib);
        auto make_saxpy = [&](const char* which) {
            return [&, which]() -> double {
                if (std::strcmp(which, "scalar") == 0) {
                    kernels::saxpy_f32_scalar(y.data(), a.data(), 1.5f, n);
                } else if (std::strcmp(which, "intrinsics") == 0) {
                    kernels::saxpy_f32_intrinsics(y.data(), a.data(), 1.5f, n);
                } else {
                    vectis_asm_saxpy_f32(y.data(), a.data(), 1.5f, n);
                }
                return static_cast<double>(y[0]);
            };
        };
        vbench::compare(title2, opt, {"scalar", "intrinsics", "asm"}, make_saxpy);
    }

    vbench::note("min of 21 runs; cycles/elem from rdtsc across the whole batch");
    vbench::total_sink();
    return 0;
}
