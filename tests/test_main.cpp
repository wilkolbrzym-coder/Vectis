// ===========================================================================
// Vectis test runner
// ===========================================================================
//
// Exit codes, so a caller can tell the three outcomes apart:
//
//   0  everything selected ran and passed (skips allowed: a tier the host does
//      not have is a missing CPU, not a defect)
//   1  at least one test failed
//   2  the command line was wrong, or the filter matched nothing
//
// Skips are reported as `skip` and counted on their own line.  They used to be
// reported as `ok`, which made "2/2 tests passed (0 checks)" - a green result
// that executed nothing - look like a pass in a CI log.
// ===========================================================================
#include "vectis_test.hpp"

#include <cstdio>
#include <cstring>

namespace {

/// Report a bad command line instead of ignoring it.  An unknown argument used
/// to be skipped silently, so `--filtr avx512` ran the whole suite and exited 0
/// while the log said a filter had been applied.
int usage(const char* arg) {
    std::printf("unknown argument: %s\n"
                "usage: vectis_tests [--filter <substring>] [--list]\n",
                arg == nullptr ? "" : arg);
    return 2;
}

} // namespace

int main(int argc, char** argv) {
    const char* filter = nullptr;
    bool list_only = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--filter") == 0) {
            if (i + 1 >= argc) return usage(argv[i]);
            filter = argv[++i];
        } else if (std::strncmp(argv[i], "--filter=", 9) == 0) {
            filter = argv[i] + 9;
        } else if (std::strcmp(argv[i], "--list") == 0) {
            list_only = true;
        } else {
            return usage(argv[i]);
        }
    }

    if (list_only) {
        for (const auto& t : vtest::registry()) std::printf("%s\n", t.name);
        return 0;
    }

    int run = 0;
    int failed = 0;
    int skipped = 0;

    for (const auto& t : vtest::registry()) {
        if (filter != nullptr && std::strstr(t.name, filter) == nullptr) continue;

        ++run;
        const int before = vtest::failure_count();
        vtest::current_test() = t.name;
        vtest::skipped() = false;
        t.fn();
        const bool ok = vtest::failure_count() == before;

        if (!ok) {
            ++failed;
            std::printf("FAIL %s\n", t.name);
        } else if (vtest::skipped()) {
            ++skipped;
            std::printf("skip %s\n", t.name);
        } else {
            std::printf("ok   %s\n", t.name);
        }
        std::fflush(stdout);
    }

    std::printf("\n%d/%d tests passed", run - failed, run);
    if (skipped != 0) std::printf(", %d skipped", skipped);
    std::printf("  (%d checks, %d failures)\n",
                vtest::check_count(), vtest::failure_count());

    if (run == 0) {
        std::printf("no tests matched filter '%s'\n", filter ? filter : "");
        return 2;
    }
    return failed == 0 ? 0 : 1;
}
