// ===========================================================================
// Vectis test runner
// ===========================================================================
#include "vectis_test.hpp"

#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    const char* filter = nullptr;
    bool list_only = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
            filter = argv[++i];
        } else if (std::strncmp(argv[i], "--filter=", 9) == 0) {
            filter = argv[i] + 9;
        } else if (std::strcmp(argv[i], "--list") == 0) {
            list_only = true;
        }
    }

    if (list_only) {
        for (const auto& t : vtest::registry()) std::printf("%s\n", t.name);
        return 0;
    }

    int run = 0;
    int failed = 0;
    for (const auto& t : vtest::registry()) {
        if (filter != nullptr && std::strstr(t.name, filter) == nullptr) continue;

        ++run;
        const int before = vtest::failure_count();
        vtest::current_test() = t.name;
        t.fn();
        const bool ok = vtest::failure_count() == before;

        std::printf("%-4s %s\n", ok ? "ok" : "FAIL", t.name);
        std::fflush(stdout);
        if (!ok) ++failed;
    }

    std::printf("\n%d/%d tests passed  (%d checks, %d failures)\n",
                run - failed, run, vtest::check_count(), vtest::failure_count());

    if (run == 0) {
        std::printf("no tests matched filter '%s'\n", filter ? filter : "");
        return 2;
    }
    return failed == 0 ? 0 : 1;
}
