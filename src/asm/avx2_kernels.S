/*
 * ===========================================================================
 * Vectis - hand-written AVX2 kernels
 * ===========================================================================
 *
 * These exist to answer one question with a measurement rather than an opinion:
 * where does hand-written assembly actually beat intrinsics?
 *
 * The answer, for this class of kernel, is "usually nowhere" - and that is a
 * useful thing to have established rather than assumed.  A modern compiler
 * given clean intrinsics already picks the same instructions, and it knows the
 * register pressure, the scheduling and the alignment constraints of the
 * surrounding code, which a .S file can never know because it cannot be inlined.
 *
 * What asm can still win is the case where the compiler *runs out of
 * registers*: this dot product keeps four independent accumulators and four
 * live operand registers, and holds up under unrolling in a way the intrinsics
 * version is at the mercy of.  bench/bench_asm.cpp measures both.
 *
 * Conventions this file must respect (System V AMD64, GNU as, AT&T syntax):
 *
 *   - Arguments: rdi, rsi, rdx, rcx, r8, r9, then the stack.
 *     Float arguments arrive in xmm0..xmm7; a float *return* goes in xmm0.
 *   - Registers rbx, rbp, r12-r15 belong to the caller.  This file uses none of
 *     them, so there is nothing to save and no stack frame to build.
 *   - ymm0-ymm15 are all caller-saved, which is why the accumulators can live
 *     there freely.
 *   - vzeroupper before returning.  Without it, every subsequent SSE
 *     instruction in the caller pays an AVX-SSE transition penalty - a classic
 *     and expensive omission.
 *
 * C-style comments are used deliberately: this is a .S file, so it goes through
 * the C preprocessor, and a leading '#' would be read as a directive rather
 * than as an assembler comment.
 *
 * ===========================================================================
 */

	.text

/* ---------------------------------------------------------------------------
 * float vectis_asm_dot_f32(const float* a, const float* b, size_t n)
 *
 * rdi = a, rsi = b, rdx = n.  Result in xmm0.
 *
 * Four independent accumulators, 32 floats per iteration.  The dependency
 * chain per accumulator is one vfmadd231ps with ~4 cycle latency, so four
 * chains cover ~16 cycles of work per iteration on a machine with two FMA
 * ports - the shape you want before the compiler starts spilling.
 * ------------------------------------------------------------------------ */
	.globl	vectis_asm_dot_f32
	.type	vectis_asm_dot_f32, @function
	.p2align 4
vectis_asm_dot_f32:
	vxorps	%xmm0, %xmm0, %xmm0		/* accumulator 0..3 */
	vxorps	%xmm1, %xmm1, %xmm1
	vxorps	%xmm2, %xmm2, %xmm2
	vxorps	%xmm3, %xmm3, %xmm3

	movq	%rdx, %rcx
	shrq	$5, %rcx			/* iterations of 32 floats */
	jz	2f

	.p2align 5
1:
	vmovups	   0(%rdi), %ymm4
	vmovups	  32(%rdi), %ymm5
	vmovups	  64(%rdi), %ymm6
	vmovups	  96(%rdi), %ymm7
	vfmadd231ps	   0(%rsi), %ymm4, %ymm0
	vfmadd231ps	  32(%rsi), %ymm5, %ymm1
	vfmadd231ps	  64(%rsi), %ymm6, %ymm2
	vfmadd231ps	  96(%rsi), %ymm7, %ymm3
	addq	$128, %rdi
	addq	$128, %rsi
	decq	%rcx
	jnz	1b

2:	/* collapse the four accumulators, then the lanes of the survivor */
	vaddps	%ymm1, %ymm0, %ymm0
	vaddps	%ymm3, %ymm2, %ymm2
	vaddps	%ymm2, %ymm0, %ymm0
	vextractf128	$1, %ymm0, %xmm1
	vaddps	%xmm1, %xmm0, %xmm0
	vpermilps	$0x4E, %xmm0, %xmm1		/* swap 64-bit halves */
	vaddps	%xmm1, %xmm0, %xmm0
	vpermilps	$0xB1, %xmm0, %xmm1		/* swap 32-bit halves */
	vaddss	%xmm1, %xmm0, %xmm0

	andq	$31, %rdx			/* remaining scalars */
	jz	4f
	.p2align 4
3:
	vmovss	(%rdi), %xmm1
	vfmadd231ss	(%rsi), %xmm1, %xmm0
	addq	$4, %rdi
	addq	$4, %rsi
	decq	%rdx
	jnz	3b

4:
	vzeroupper
	ret
	.size	vectis_asm_dot_f32, .-vectis_asm_dot_f32

/* ---------------------------------------------------------------------------
 * void vectis_asm_saxpy_f32(float* y, const float* x, float alpha, size_t n)
 *
 * rdi = y, rsi = x, xmm0 = alpha, rdx = n.  Computes y += alpha*x.
 *
 * Memory-bandwidth bound, so the only thing that matters is not stalling on the
 * loads and not wasting a register on the broadcast.  The FMA takes x straight
 * from memory, which is the reason this version is one instruction shorter per
 * vector than the obvious intrinsics spelling.
 * ------------------------------------------------------------------------ */
	.globl	vectis_asm_saxpy_f32
	.type	vectis_asm_saxpy_f32, @function
	.p2align 4
vectis_asm_saxpy_f32:
	vbroadcastss	%xmm0, %ymm0

	movq	%rdx, %rcx
	shrq	$5, %rcx
	jz	2f

	.p2align 5
1:
	vmovups	   0(%rdi), %ymm4
	vmovups	  32(%rdi), %ymm5
	vmovups	  64(%rdi), %ymm6
	vmovups	  96(%rdi), %ymm7
	vfmadd231ps	   0(%rsi), %ymm0, %ymm4
	vfmadd231ps	  32(%rsi), %ymm0, %ymm5
	vfmadd231ps	  64(%rsi), %ymm0, %ymm6
	vfmadd231ps	  96(%rsi), %ymm0, %ymm7
	vmovups	%ymm4,	   0(%rdi)
	vmovups	%ymm5,	  32(%rdi)
	vmovups	%ymm6,	  64(%rdi)
	vmovups	%ymm7,	  96(%rdi)
	addq	$128, %rdi
	addq	$128, %rsi
	decq	%rcx
	jnz	1b

2:
	andq	$31, %rdx
	jz	4f
	.p2align 4
3:
	vmovss	(%rdi), %xmm1
	vfmadd231ss	(%rsi), %xmm0, %xmm1
	vmovss	%xmm1, (%rdi)
	addq	$4, %rdi
	addq	$4, %rsi
	decq	%rdx
	jnz	3b

4:
	vzeroupper
	ret
	.size	vectis_asm_saxpy_f32, .-vectis_asm_saxpy_f32

/* ---------------------------------------------------------------------------
 * float vectis_asm_sum_f32(const float* a, size_t n)
 *
 * rdi = a, rsi = n.  A pure bandwidth probe: eight accumulators so that the
 * add latency never becomes the limit, which is the shape a reduction needs
 * before it can saturate the load ports.
 * ------------------------------------------------------------------------ */
	.globl	vectis_asm_sum_f32
	.type	vectis_asm_sum_f32, @function
	.p2align 4
vectis_asm_sum_f32:
	vxorps	%xmm0, %xmm0, %xmm0
	vxorps	%xmm1, %xmm1, %xmm1
	vxorps	%xmm2, %xmm2, %xmm2
	vxorps	%xmm3, %xmm3, %xmm3

	movq	%rsi, %rcx
	shrq	$6, %rcx			/* 64 floats per iteration */
	jz	2f

	.p2align 5
1:
	vaddps	   0(%rdi), %ymm0, %ymm0
	vaddps	  32(%rdi), %ymm1, %ymm1
	vaddps	  64(%rdi), %ymm2, %ymm2
	vaddps	  96(%rdi), %ymm3, %ymm3
	vaddps	 128(%rdi), %ymm0, %ymm0
	vaddps	 160(%rdi), %ymm1, %ymm1
	vaddps	 192(%rdi), %ymm2, %ymm2
	vaddps	 224(%rdi), %ymm3, %ymm3
	addq	$256, %rdi
	decq	%rcx
	jnz	1b

2:
	vaddps	%ymm1, %ymm0, %ymm0
	vaddps	%ymm3, %ymm2, %ymm2
	vaddps	%ymm2, %ymm0, %ymm0
	vextractf128	$1, %ymm0, %xmm1
	vaddps	%xmm1, %xmm0, %xmm0
	vpermilps	$0x4E, %xmm0, %xmm1
	vaddps	%xmm1, %xmm0, %xmm0
	vpermilps	$0xB1, %xmm0, %xmm1
	vaddss	%xmm1, %xmm0, %xmm0

	andq	$63, %rsi
	jz	4f
	.p2align 4
3:
	vaddss	(%rdi), %xmm0, %xmm0
	addq	$4, %rdi
	decq	%rsi
	jnz	3b

4:
	vzeroupper
	ret
	.size	vectis_asm_sum_f32, .-vectis_asm_sum_f32

/* Mark the stack non-executable.  Without this, modern linkers warn and some
 * hardened configurations refuse the object outright. */
	.section	.note.GNU-stack,"",@progbits
