# Vectis

[![CI](https://github.com/wilkolbrzym-coder/Vectis/actions/workflows/ci.yml/badge.svg)](https://github.com/wilkolbrzym-coder/Vectis/actions/workflows/ci.yml)
[![Quality](https://github.com/wilkolbrzym-coder/Vectis/actions/workflows/quality.yml/badge.svg)](https://github.com/wilkolbrzym-coder/Vectis/actions/workflows/quality.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A compile-time SIMD engine for C++ — a header-only vector library that maps one
source text onto 128/256/512-bit hardware, plus an optional hand-written
assembly layer that exists mainly to answer whether it earns its keep.

**Vectis is a C++20/23 library that is *ready* for C++26, not a C++26 library.**
That distinction is the most important thing on this page and is explained under
[The C++26 question](#the-c26-question) below.

---

## What it does

```cpp
#include <vectis/simd/vec.hpp>
#include <vectis/simd/math.hpp>

using namespace vectis;

// One line, three machines:
using v16 = f32<16>;      // 1 ZMM on AVX-512, 2 YMM on AVX2, 16 floats scalar
static_assert(v16::num_regs == 16 / native_width_v<float>);

auto a = v16::load(a_ptr);
auto b = v16::load(b_ptr);
math::rsqrt(a * b + a).store(out_ptr);
```

```cpp
#include <vectis/soa/soa.hpp>

struct Particle { float x, y, z, vx, vy, vz; };

using F_x = field_of<&Particle::x>;     // no macro, no code generator
using L   = layout<Particle, &Particle::x, &Particle::y, &Particle::z,
                            &Particle::vx, &Particle::vy, &Particle::vz>;

soa_array<L> particles(aos_ptr, count);          // AoS -> SoA

particles.for_each_block<16>([&](std::size_t i) { // vectorised kernel over it
    particles.store<F_x, 16>(i, particles.load<F_x, 16>(i) +
                                 particles.load<F_vx, 16>(i) * dt);
});
```

---

## What works, verified

| | Status |
|---|---|
| Scalar / AVX2 / AVX-512 backends for `f32`, `f64`, `i32`, `u32`, `i64`, `u64` | working |
| Multi-register vectors, partial-register tails, padded lanes | working, canary-tested |
| `std::simd`-shaped API: masks, `select`, `all/any/none/bits` | working |
| AoS → SoA without reflection, via pointer-to-member NTTPs | working |
| Assembly layer (GAS `.S`, AVX2): `dot`, `saxpy`, `sum` | working, bit-parity tested |
| Tests | 23 tests, ~71 000 assertions, green |
| AVX-512 **execution** | **not verified here** — see below |

### The hardware this was built on

The development machine is an **Intel i5-6500T (Skylake, 2015)**: AVX2, FMA,
BMI2, AES, PCLMULQDQ — and **no AVX-512 at all**. So:

* AVX2 is the ceiling for anything measured in this repository.
* The AVX-512 backend is **compiled and linked** and its code generation is
  verified (it emits `zmm` registers and `vfmadd132ps`), but it **cannot be
  executed here**. Running a tier-`avx512` binary on this CPU dies with SIGILL,
  which is the expected result and not a bug.
* Executing the AVX-512 tests is what CI is for: GitHub's `ubuntu` runners are
  usually Xeon Platinum (Ice Lake-SP) and have the full AVX-512 quartet plus
  VBMI2, VNNI and IFMA.

The working set is also small — 32 KiB L1d, 256 KiB L2, 6 MiB L3 — and the
governor is `powersave`, so absolute timings below are noisy and optimistic
comparisons should be read as ratios, not as benchmarks of your hardware.

---

## The C++26 question, answered honestly

The project brief asks for `std::experimental::simd`, C++26 reflection, and
concepts. Here is what was actually available, checked rather than assumed:

| Feature | Status on GCC 15.2 | Consequence |
|---|---|---|
| `std::simd` (P1928, C++26) | **absent** — no `<simd>`, no `__cpp_lib_simd` | Vectis ships its own vector type |
| `<experimental/simd>` (TS) | **absent** from this libstdc++ | — |
| Reflection (P2996) | **absent** — no `__cpp_lib_meta`, no `__cpp_impl_reflection` | AoS→SoA uses pointer-to-member NTTPs instead |
| Concepts (C++20) | available | used pervasively, and load-bearing |
| `-std=c++26` | accepted, reports `__cplusplus = 202400` | so "C++26" here means "the C++2c working draft", not the published standard |

Vectis therefore does **not** depend on `std::simd` or reflection. It does not
merely degrade without them; it does not need them. The vector type is built
directly on intrinsics, which is also why it can do things a `std::simd` wrapper
could not — the mask representation, the capability concepts, and the
partial-register model are all design decisions this library gets to make.

Both are written so that the better facility slots in when it arrives: a
`std::simd` backend is one more ABI tag, and reflection replaces `field_of`
without the container noticing.

---

## Design decisions worth knowing

**Vectors are N lanes, not one register.** `f32<16>` is two YMMs on AVX2 and one
ZMM on AVX-512, and every kernel written against it works on both. Lane counts
that do not fill the last register are allowed and padded: `f32<6>` on AVX2 is
one YMM with two dead lanes and exactly six observable ones. Loads, stores,
`to_array`, `bits()` and every reduction respect that boundary — a mask over 5
lanes reports 5 bits, never the 8 the register holds, and there is a test that
fails if it ever does.

**The scalar backend is the correctness oracle.** It is written with plain C++
operators that deliberately mirror hardware semantics rather than `<cmath>`:
`min(a,b)` is `(a<b)?a:b`, not `std::fmin`, because they disagree on NaN and on
the sign of zero. Every vector backend is compared against it lane by lane, bit
pattern by bit pattern.

**Capability concepts instead of silent fallbacks.** AVX2 has no 64-bit
multiply, no 64-bit min/max and no ordered 64-bit compare. A kernel constrained
on `BackendHasMul<int64_t, avx2_abi>` simply does not exist there — a compile
error at the call site, not a scalar emulation that quietly runs at a quarter
speed.

**No sse42 tier.** A 128-bit backend would be a third parallel implementation of
every operation, to serve hardware that in practice always has AVX2 (Haswell,
2013). The multi-register path is already exercised by wide vectors at the
scalar and AVX2 tiers, so the coverage is not lost, only the duplication.

---

## The assembly verdict

The brief asked for hand-written assembly "squeezing the absolute physical
maximum" out of the SIMD generation. Three kernels were written in GAS `.S`
(`src/asm/avx2_kernels.S`), each with an intrinsics twin built to the *same*
shape — four accumulators, 32 floats per iteration — so the comparison isolates
the one variable that matters.

`dot_f32`, n = 1024 floats, L1-resident, min of 21 runs:

| kernel | ns | cycles/element | speedup |
|---|---|---|---|
| scalar, 1 accumulator | 1491 | 4.92 | 1.00x |
| scalar, 4 accumulators | 679 | 1.73 | 2.05x |
| intrinsics | 107 | 0.338 | 13.93x |
| **assembly** | **107** | **0.333** | **13.93x** |

**Assembly ties intrinsics exactly.** The generated inner loops are
instruction-for-instruction identical — both compile to four `vmovups` /
`vfmadd231ps` / four `vmovups` per 32 elements. Where assembly appeared to win
by 1.3x (`saxpy` at n=1024), the cause was not better code: the C++ version was
missing `restrict`, so GCC inserted runtime alias checks and a peeled prologue.
Adding `restrict` moved it. At DRAM-resident sizes the gap vanishes entirely and
assembly sometimes *loses* (0.92x).

The bigger finding is in the table's first two rows. **The 13.9x speedup is not
about vector width.** A plain scalar loop with four independent accumulators —
still IEEE-exact, still no `-ffast-math`, just a different summation order the
programmer chose — already captures 2.05x. Compiling the naive loop with
`-ffast-math` gets 6.9x, leaving under 2.3x attributable to width and FMA.

That is the real value proposition of an explicit SIMD engine: **it is not that
the lanes are wider, it is that you are allowed to choose the reduction order,
portably and explicitly.** A compiler cannot reassociate your floating-point
sum without a flag that also turns off IEEE semantics everywhere else in the
translation unit.

**Recommendation: do not add assembly by default.** It costs a second
implementation of every kernel, a bit-parity test to keep the two honest, and an
ABI boundary that cannot be inlined — and the measurements here show no return.
Keep the layer, keep the parity test, and reach for it only when a profile shows
a specific kernel where the compiler's register allocation is the problem.

## The SoA verdict

`bench_soa.cpp`, particle integration over 7 fields, AoS scalar against SoA
vectorised:

| working set | AoS scalar | SoA n=8 | SoA n=16 | AoS→SoA conversion |
|---|---|---|---|---|
| 28 KiB (L1) | 1.00x | 1.61x | 1.37x | 1.00x |
| 448 KiB (L2/L3) | 1.00x | 1.41x | 1.28x | 0.49x |
| 7 MiB (L3) | 1.00x | 0.95x | 0.96x | 0.23x |
| 114 MiB (DRAM) | 1.00x | 0.95x | 0.87x | 0.24x |

Two conclusions, both unwelcome if you were hoping for a blanket win:

1. **SoA pays only while the data is L1/L2-resident** (1.3–1.6x). Past the last
   cache level everything is bandwidth-bound, and AoS — one stream instead of
   seven — ties or beats it.
2. **The conversion costs 2–4x a kernel pass** at large sizes, because a strided
   gather is inherently slower than a linear sweep. Converting inside the hot
   loop is a pessimisation. It only pays when the data *stays* SoA across
   several kernels, which is what `soa_array` is for: it owns storage rather
   than being a view.

(An earlier version of this benchmark reported conversion at 127 cycles per
element. That was measuring seven `malloc`/`free` pairs per call, not the
conversion. The harness now pre-allocates, and the note is kept because a
measurement artefact that absurd is worth remembering.)

---

## Building

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVECTIS_ISA_TIER=avx2
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/tools/vectis-cpuinfo
```

One knob controls the ISA tier, because the two things it must control — the
`-m` flags and the ceiling the library may emit — must never disagree:

| `VECTIS_ISA_TIER` | Flags | Meaning |
|---|---|---|
| `auto` (default) | `-march=x86-64-v3` | portable baseline; AVX-512 reachable only via runtime-checked target attributes |
| `native` | `-march=native` | fastest this exact CPU can do |
| `scalar` | `-mno-avx -mno-avx2 -mno-fma` | reference path, the correctness oracle |
| `avx2` | `-march=x86-64-v3 -mno-avx512*` | 256-bit ceiling |
| `avx512` | `-march=x86-64-v4` | 512-bit ceiling; needs AVX-512 hardware to *run* |

Benchmarks: `./build/bench/bench_simd`, `bench_soa`, `bench_asm` (add `--quick`
for CI-sized runs).

### Using it from another project

```cmake
find_package(vectis 0.1 REQUIRED)
target_link_libraries(my_app PRIVATE vectis::vectis)
```

The package supplies the include path and a **C++20 floor**, and nothing else.
In particular it does *not* impose `-march` or any warning flags on you: those
are your build's decisions, and a library that overrides them is a library
people work around.

The consequence is worth stating plainly, because it surprises people: a
consumer that passes no `-march` at all gets the **scalar** backend, since
`native_abi` follows whatever the compiler was told. That is correct but slow.
Pass `-march=x86-64-v3` (AVX2, universal since Haswell) or
`-DVECTIS_ISA_TIER` in your own build to get vector code. `vectis-cpuinfo`
prints which tier a binary ended up with.

### Continuous integration

Two workflows, with different jobs:

* **`ci.yml`** — the gate, on every push and pull request. ISA tiers, six
  compilers, three language standards, six optimisation levels, standalone
  header compilation across two compilers × three standards × three tiers,
  an install-and-`find_package` consumer test, assembly parity with the layer
  both on and off, and aarch64 plus two macOS runners. Every job names what it
  defends against. A final `gate` job aggregates them for branch protection.
* **`quality.yml`** — the deeper pass, on pull requests and weekly. ASan+UBSan,
  TSan, `_GLIBCXX_DEBUG`, `_GLIBCXX_ASSERTIONS`, `_FORTIFY_SOURCE=3`, clang-tidy,
  cppcheck, GCC's `-fanalyzer`, coverage, and a benchmark smoke run.

The rule throughout: **build always, run when the hardware allows.** A tier
that cannot execute on a runner is still compiled, because a compile error is a
bug; only the execution is skipped, and the job says so in its summary rather
than passing quietly. The `hardware` job reports the runner's CPU first,
because every performance claim in this README is meaningless without it.

Steps that depend on a tool which was unavailable on the development machine —
valgrind, clang-tidy, cppcheck — are marked `continue-on-error: true` and say so
in their comments. A check nobody has seen pass is not a gate; it is a red
herring that teaches people to ignore CI. Promote them once they have run green.

### Test layout, and why it is split

Three translation units exist for the backends, and that is not an accident:
GCC declares `_mm512_cmplt_ps_mask` with a *different, reduced signature* when
AVX512F is disabled, so merely mentioning it — inside a function nobody calls —
is a hard compile error. The AVX-512 backend therefore cannot sit unexecuted
inside an AVX2 binary; it needs a TU compiled for it and entered behind
`cpu::has(isa_level::avx512)`.

The payoff is that **one test binary exercises all three backends on any host**,
whatever tier the library was configured for. A scalar-tier build still proves
the AVX2 and AVX-512 kernels are correct; it just cannot run all of them, and it
says so loudly instead of skipping silently.

---

## Repository layout

```
include/vectis/
  core/config.hpp        compiler / arch / ISA detection, target attributes
  core/cpu.hpp           runtime CPUID feature detection
  core/concepts.hpp      SimdVec, SimdMask, Field, SoaLayout
  simd/abi.hpp           register-file strategy tags
  simd/backend.hpp       the primitive protocol + capability concepts
  simd/backend_scalar.hpp  reference implementation (the oracle)
  simd/backend_avx2.hpp    256-bit, including its documented gaps
  simd/backend_avx512.hpp 512-bit, with real mask registers
  simd/vec.hpp           basic_vec<T, N, Abi> — multi-register folds
  simd/mask.hpp          per-lane predicates, padding-safe
  simd/math.hpp          rsqrt/rcp (refined), clamp/lerp/saturate/smoothstep
  simd/reduce.hpp        reductions, dot/length/normalize, chunked iteration
  soa/soa.hpp            AoS -> SoA via pointer-to-member descriptors
  kernels/asm_kernels.hpp  asm declarations + intrinsics twins
src/asm/avx2_kernels.S   hand-written kernels (GAS, AT&T syntax)
tests/                   23 tests; battery shared across three TUs
bench/                   asm-vs-intrinsics, vector-vs-scalar, SoA
tools/cpuinfo.cpp        what this binary will use, and what the CPU can run
```

---

## Known limitations

* **AVX-512 execution is unverified on the development machine.** Code
  generation is verified; behaviour is CI's job. Until a CI run goes green on a
  Xeon runner, treat the AVX-512 backend as compiled-but-unproven.
* **No `exp`/`log`/`sin`/`cos`.** These need integer↔float lane conversions and
  a round-to-nearest in the backend protocol, plus range reduction. Doing them
  badly is worse than not doing them; they are a deliberate omission, not an
  oversight.
* **Reductions stage through memory.** A shuffle tree would avoid the
  store-forwarding stall, at the cost of a different shuffle per lane width and
  element type. It is measured in `bench_simd` rather than assumed.
* **`iota(first, step)` requires a backend multiply**, so it does not exist for
  64-bit lanes on AVX2. Constrained rather than emulated.
* **No gather/scatter for strided access**, no `int8`/`int16` lanes, no
  compress/expand — all present on AVX-512 and all still on the table.
* **No ARM NEON backend.** The scalar path carries aarch64 — and CI checks that
  it does — but there is no vector backend there yet.

## Next steps, in the order they would pay off

1. Get CI green on an AVX-512 runner. Nothing below matters until the widest
   backend has actually executed somewhere.
2. Add `int8`/`int16` lanes with `vpermb`/`vpshufb` — the crypto and compression
   workloads in the original brief need them, and byte permutes are where
   AVX-512's advantage over AVX2 is largest.
3. A shuffle-tree reduction, benchmarked against the current one.
4. An ARM NEON backend, now that the protocol and its capability concepts are
   settled.
