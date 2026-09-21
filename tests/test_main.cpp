// ===========================================================================
// Vectis test runner
// ===========================================================================
//
// Exit codes, so a caller can tell the three outcomes apart:
//
//   0  everything selected ran and passed (skips allowed: a tier the host does
//      not have is a missing CPU, not a defect)
//   1  at least one test failed, or --fail-on-skip was given and something
//      skipped
//   2  the command line was wrong, or the filter matched nothing
//
// Skips are reported as `skip` and counted on their own line.  They used to be
// reported as `ok`, which made "2/2 tests passed (0 checks)" - a green result
// that executed nothing - look like a pass in a CI log.
//
// Counting them separately is only half of the fix, because the exit code is
// what a caller reads: a job that means "verify AVX-512" and runs a suite that
// skipped every test still exits 0 without --fail-on-skip, so a green ctest
// cannot tell "the backend ran and passed" from "the backend never ran".  That
// is a distinction two CI gates depend on - the AVX-512 battery and the
// reflection path - and both now pass this flag rather than trusting the log.
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
                "usage: vectis_tests [--filter <substring>] [--list] [--fail-on-skip]\n",
                arg == nullptr ? "" : arg);
    return 2;
}

} // namespace

int main(int argc, char** argv) {
    const char* filter = nullptr;
    bool list_only = false;
    bool fail_on_skip = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--filter") == 0) {
            if (i + 1 >= argc) return usage(argv[i]);
            filter = argv[++i];
        } else if (std::strncmp(argv[i], "--filter=", 9) == 0) {
            filter = argv[i] + 9;
        } else if (std::strcmp(argv[i], "--list") == 0) {
            list_only = true;
        } else if (std::strcmp(argv[i], "--fail-on-skip") == 0) {
            fail_on_skip = true;
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

    // The ratio counts only the tests that executed.  "25/25 passed, 3 skipped"
    // reads as 25 successes when 22 of them ran, which is the same
    // green-by-omission this runner was fixed for once already.
    const int executed = run - skipped;
    std::printf("\n%d/%d tests passed", executed - failed, executed);
    if (skipped != 0) std::printf(", %d skipped", skipped);
    std::printf("  (%d checks, %d failures)\n",
                vtest::check_count(), vtest::failure_count());

    if (run == 0) {
        std::printf("no tests matched filter '%s'\n", filter ? filter : "");
        return 2;
    }
    if (fail_on_skip && skipped != 0) {
        std::printf("--fail-on-skip: %d of %d selected tests did not execute here\n",
                    skipped, run);
        return 1;
    }
    return failed == 0 ? 0 : 1;
}
