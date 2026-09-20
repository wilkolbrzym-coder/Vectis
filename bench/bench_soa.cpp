// ===========================================================================
// Vectis - does AoS to SoA actually pay?
// ===========================================================================
//
// The claim under test: converting AoS to SoA costs a memory pass, so it is
// only worth it when the data stays SoA across several passes.  This measures
// both shapes over a range of working-set sizes.
//
// The conversion column is the honest part of the answer.  If AoS-to-SoA costs
// as much as the kernel it accelerates, then for a single pass it is a loss -
// and the number will say so.
//
// ===========================================================================
#include "bench_harness.hpp"

#include <vectis/core/cpu.hpp>
#include <vectis/soa/soa.hpp>
#include <vectis/simd/vec.hpp>

#include <cstdio>
#include <cstring>
#include <vector>

using namespace vectis;

namespace {

struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float life;
};

using F_x    = field_of<&Particle::x>;
using F_y    = field_of<&Particle::y>;
using F_z    = field_of<&Particle::z>;
using F_vx   = field_of<&Particle::vx>;
using F_vy   = field_of<&Particle::vy>;
using F_vz   = field_of<&Particle::vz>;
using F_life = field_of<&Particle::life>;

using Layout = mem_layout<Particle, &Particle::x, &Particle::y, &Particle::z,
                      &Particle::vx, &Particle::vy, &Particle::vz,
                      &Particle::life>;

std::vector<Particle> make_particles(std::size_t n) {
    std::vector<Particle> p(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float f = static_cast<float>(i);
        p[i] = Particle{f * 0.37f - 5.0f, f * -0.11f + 2.0f, f * 0.05f,
                        f * 0.01f + 0.3f,  f * -0.02f,        f * 0.004f - 0.1f,
                        1.0f - f * 0.001f};
    }
    return p;
}

void integrate_aos(Particle* p, std::size_t n, float dt, float damping) {
    for (std::size_t i = 0; i < n; ++i) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
        p[i].vx *= damping;
        p[i].vy *= damping;
        p[i].vz *= damping;
        p[i].life -= dt;
    }
}

template <std::size_t N>
void integrate_soa(soa_array<Layout>& a, float dt, float damping) {
    using V = basic_vec<float, N>;
    const V vdt = V::set1(dt);
    const V vdamp = V::set1(damping);
    const std::size_t done = a.for_each_block<N>([&](std::size_t i) {
        a.store<F_x, N>(i, a.load<F_x, N>(i) + a.load<F_vx, N>(i) * vdt);
        a.store<F_y, N>(i, a.load<F_y, N>(i) + a.load<F_vy, N>(i) * vdt);
        a.store<F_z, N>(i, a.load<F_z, N>(i) + a.load<F_vz, N>(i) * vdt);
        a.store<F_vx, N>(i, a.load<F_vx, N>(i) * vdamp);
        a.store<F_vy, N>(i, a.load<F_vy, N>(i) * vdamp);
        a.store<F_vz, N>(i, a.load<F_vz, N>(i) * vdamp);
        a.store<F_life, N>(i, a.load<F_life, N>(i) - vdt);
    });
    auto* px = a.data<F_x>(); auto* py = a.data<F_y>(); auto* pz = a.data<F_z>();
    auto* vx = a.data<F_vx>(); auto* vy = a.data<F_vy>(); auto* vz = a.data<F_vz>();
    auto* lf = a.data<F_life>();
    for (std::size_t i = done; i < a.size(); ++i) {
        px[i] += vx[i] * dt; py[i] += vy[i] * dt; pz[i] += vz[i] * dt;
        vx[i] *= damping;    vy[i] *= damping;    vz[i] *= damping;
        lf[i] -= dt;
    }
}

} // namespace

int main(int argc, char** argv) {
    bool quick = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quick") == 0) quick = true;
    }

    std::printf("\nvectis benchmark: SoA transform\n");
    std::printf("===============================\n");
    {
        const auto b = cpu::brand();
        std::printf("  cpu  : %.*s\n", static_cast<int>(b.size()), b.data());
    }
    std::printf("  tier : %s\n", to_string(compiled_isa));

    constexpr float dt = 0.016f;
    constexpr float damping = 0.999f;
    constexpr std::size_t stride = 16;   // lanes per block in the SoA kernel

    const std::vector<std::size_t> sizes =
        quick ? std::vector<std::size_t>{4096, 262144}
              : std::vector<std::size_t>{1024, 16384, 262144, 4194304};

    for (const std::size_t n : sizes) {
        const auto src = make_particles(n);
        std::vector<Particle> aos = src;
        soa_array<Layout> soa(src.data(), n);
        soa_array<Layout> conv(n);      // pre-allocated: see the note below

        const double kib = static_cast<double>(n) * sizeof(Particle) / 1024.0;
        char title[192];
        std::snprintf(title, sizeof title,
                      "particle integrate: n=%zu (%.0f KiB AoS, %.0f KiB touched)",
                      n, kib, kib * 2.0);

        vbench::options opt;
        opt.elems = n;
        opt.bytes = n * sizeof(Particle) * 2;
        opt.reps  = quick ? 5 : 21;

        auto make = [&](const char* which) {
            return [&, which]() -> double {
                if (std::strcmp(which, "AoS scalar") == 0) {
                    integrate_aos(aos.data(), n, dt, damping);
                    return static_cast<double>(aos[0].x);
                }
                if (std::strcmp(which, "SoA SIMD n=16") == 0) {
                    integrate_soa<stride>(soa, dt, damping);
                    return static_cast<double>(soa.data<F_x>()[0]);
                }
                if (std::strcmp(which, "SoA SIMD n=8") == 0) {
                    integrate_soa<8>(soa, dt, damping);
                    return static_cast<double>(soa.data<F_x>()[0]);
                }
                // The conversion on its own: what SoA costs to enter.
                //
                // `assign` on storage that already exists, NOT the
                // constructor: measuring the allocating constructor would
                // report seven malloc/free pairs per call as if it were
                // conversion cost, which is a measurement artefact and a
                // wildly misleading one (it showed 127 cycles per element).
                conv.assign(src.data(), n);
                return static_cast<double>(conv.data<F_x>()[0]);
            };
        };

        vbench::compare(title, opt,
                        {"AoS scalar", "SoA SIMD n=8", "SoA SIMD n=16",
                         "AoS->SoA convert"}, make);
    }

    vbench::note("the conversion column is the price of admission; it only pays "
                 "if the data stays SoA across several kernels");
    vbench::note("AoS struct is 28 bytes, so lanes are not naturally aligned");
    vbench::total_sink();
    return 0;
}
