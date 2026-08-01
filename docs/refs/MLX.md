# MLX Study Guide

This note is a map for studying the `refs/mlx/` subtree inside this repository.
It complements [`MAGENETRON.md`](MAGENETRON.md): Magnetron is the small,
readable systems-runtime reference; MLX is the "how does a serious Apple GPU
framework actually get performance" reference. Read Magnetron for runtime
shape. Read MLX for kernel shape.

For the kernel folder specifically (the part that looks intimidating), see the
companion notes:

- [`MLX_KERNELS.md`](MLX_KERNELS.md) - how `mlx/backend/metal/kernels/` is
  organized: JIT vs AOT, the `instantiate_kernel` macro, kernel naming, and the
  elementwise (unary/binary/ternary) dispatch path end to end.
- [`MLX_REDUCTION.md`](MLX_REDUCTION.md) - a focused walkthrough of MLX's
  reduction kernels, written specifically to contrast with a naive
  threadgroup-memory tree reduction (Module 5 territory).

---

## What MLX Is

MLX is a production-grade array framework for Apple silicon with three
distinguishing design choices that comtam does not (yet, or ever) need to
copy wholesale:

1. **Lazy, compiled execution.** Ops build a graph of `array` + `Primitive`
   nodes. Nothing runs until `eval()` walks the tape and schedules work. This
   is the opposite of comtam's eager-first course thesis.
2. **Unified memory as a first-class idea.** Arrays live in memory the CPU and
   GPU both address; MLX exploits that to avoid explicit host/device copies.
3. **A real kernel-generation pipeline.** Metal kernels are not just
   hand-written `.metal` files. MLX has a build-time *and* a run-time code
   generation path for the same kernel templates, plus a runtime JIT compiler
   for fused elementwise graphs.

That third point is why the kernels folder looks complicated: it is not one
mechanism, it is at least three, layered on top of each other.

---

## Architecture In One Diagram

```text
Python / C++ user API (mlx.core ops)
        |
        v
array + Primitive graph (lazy, untyped until eval)
        |
        v
mlx::core::eval(...)  -- walks the tape, topologically schedules
        |
        v
Primitive::eval_gpu(inputs, outputs)   (or eval_cpu)
        |
        v
backend/metal/{unary,binary,ternary,reduce,matmul,...}.cpp
  - classify layout (contiguous / broadcast / general)
  - collapse contiguous dims
  - build a kernel NAME string encoding op+dtype+layout+tuning
        |
        v
backend/metal/kernels.h  ->  get_*_kernel(...)
        |
        +-- AOT (default): metal::Device::get_kernel(name) against
        |     the single precompiled mlx.metallib
        |
        +-- JIT (MLX_METAL_JIT=ON): assemble MSL source string from
              embedded headers + explicit template instantiation,
              compile via MTLDevice::newLibraryWithSource, cache
        |
        v
MTL::ComputeCommandEncoder
  setComputePipelineState / setBuffer / setBytes / dispatchThreads
```

The important point: the *host-side C++* decides everything about which
kernel variant to run (contiguous vs strided, which dtype, which tile size).
The *Metal kernel* is a dumb, generic template that only knows how to combine
whatever pointers and shape/stride metadata it was handed. This is the same
split comtam already has (`comtam/core/kernel.h` picks a pipeline by string
name; `comtam/kernels/*.metal` are generic templates keyed by `[[host_name]]`)
- MLX just has many more axes of variation to encode in that name.

---

## The Core Ideas

### 1. Arrays are lazy graph nodes, not eager tensors

Study `mlx/array.h` and `mlx/primitives.h`. An `array` is a shared handle to:

- shape/dtype/strides,
- a `Primitive` describing how it was produced,
- its input arrays (graph edges),
- and, once evaluated, a data buffer.

`mlx::core::eval(...)` (`mlx/transforms.cpp`) is where laziness ends: it walks
the input graph, and for each node whose primitive targets the GPU, enqueues
`metal::make_task(...)` which finally calls `arr.primitive().eval_gpu(...)`.

This is the opposite of comtam's and Magnetron's "op runs now" model. Read it
to understand *why* comtam is choosing the harder-to-optimize-later but
easier-to-reason-about path, not to copy it.

### 2. `Primitive::eval_gpu` is the same seam as Magnetron's `mag_dispatch`

Every op subclasses `Primitive` (or `UnaryPrimitive` for single-output ops)
and implements `eval_gpu(inputs, outputs)`. Macros like `UNARY_GPU(Exp)` and
`BINARY_GPU(Add)` (`backend/metal/unary.cpp`, `binary.cpp`) generate this
boilerplate and forward into one shared helper per op family
(`unary_op_gpu`, `binary_op_gpu`, ...). Comparing this to
`mag_op_stubs.c`'s `mag_dispatch(...)` is a good exercise: both frameworks
converge on "one generic C++ function per op family that does validation,
kernel selection, and encoding," even though one is lazy/graph-based and the
other is eager.

### 3. Kernel dispatch is a layout classification problem first, a kernel-launch problem second

Before MLX ever thinks about which Metal function to call, it classifies the
inputs:

- `BinaryOpType`: `ScalarScalar` / `ScalarVector` / `VectorScalar` /
  `VectorVector` / `General` (`backend/common/binary.h`).
- `ReductionPlan::Type`: `ContiguousAllReduce` / `ContiguousReduce` /
  `ContiguousStridedReduce` / `GeneralContiguousReduce` /
  `GeneralStridedReduce` / `GeneralReduce` (`backend/common/reduce.h`).

Only after that classification does host code build a kernel name string and
look up a pipeline. This mirrors the view/stride work comtam already did in
Module 2 - the classification step *is* the hard part, the kernel itself is
mechanical once you know which case you are in.

### 4. Kernels are templates; `.metal` files (or JIT strings) are instantiation catalogs

A `kernels/*.h` file has generic Metal Shading Language templates, e.g.:

```1:9:refs/mlx/mlx/backend/metal/kernels/unary.h
template <typename T, typename U, typename Op>
[[kernel]] void unary_v(
    device const T* in,
    device U* out,
    uint index [[thread_position_in_grid]]) {
  out[index] = Op()(in[index]);
}
```

A `kernels/*.metal` file (compiled ahead-of-time) or a runtime-assembled JIT
source string then explicitly instantiates that template per op/dtype using
the `instantiate_kernel` macro, which is just C++'s explicit template
instantiation syntax plus `[[host_name("...")]]` so host code can look the
function up by string later. See [`MLX_KERNELS.md`](MLX_KERNELS.md) for the
exact macro and why this two-file split exists.

### 5. Two build modes exist for the exact same kernel logic

`MLX_METAL_JIT` (CMake option, default `OFF`) picks between:

- **Non-JIT (AOT):** every kernel variant is compiled at build time into one
  `mlx.metallib`; runtime is a pure string-keyed lookup.
- **JIT:** kernel *source text* is embedded into the C++ binary from the same
  `.h` templates, and the exact variant needed is compiled on first use via
  `MTLDevice::newLibraryWithSource`, then cached.

This is purely a binary-size-vs-cold-start-latency tradeoff; the kernel
semantics are identical either way. comtam does not need either mechanism at
its current scale (a handful of ops), but it is worth understanding *why* a
framework eventually wants this: once you have hundreds of op x dtype x
layout combinations, "compile everything, always" and "compile on first use"
become the only two sane options.

### 6. Reductions do not use a classic threadgroup halving tree

If you take away one thing from MLX for Module 5: MLX's cross-thread
reduction primitive (`threadgroup_reduce` in
`kernels/reduction/reduce_row.h`) is simdgroup-first (`simd_sum`/`simd_max`
across each 32-lane simdgroup), then combines the (few) per-simdgroup
partials with a *second* simdgroup reduction - not a `for (stride = N/2;
stride > 0; stride /= 2)` loop over threadgroup memory. See
[`MLX_REDUCTION.md`](MLX_REDUCTION.md) for the full breakdown and how it
compares to the CUDA-style tree reduction currently sketched in
`comtam/kernels/reduction.metal`.

### 7. Matmul earns a much bigger, separate kernel library ("steel")

`kernels/steel/` exists because matmul performance depends on threadgroup
tiling, cooperative loads, and Apple's simdgroup 8x8 matrix-multiply
instructions - none of which any other op needs. `matmul.cpp` routes
matrix-vector problems to simpler `gemv` kernels and matrix-matrix problems to
`steel`. This is explicitly out of scope for comtam's near-term matmul work;
a naive tiled matmul with threadgroup memory is the correct v1. Steel is the
"what performance work looks like three years later" reference, not a
template to imitate now.

---

## Concrete Files Worth Studying Closely

If you only read a handful of files, read these:

1. `mlx/array.h` / `mlx/primitives.h` - the lazy graph node and op interface.
2. `mlx/transforms.cpp` - where laziness ends (`eval`).
3. `mlx/backend/metal/metal.cpp` - `make_task`, where `eval_gpu` gets called.
4. `mlx/backend/metal/unary.cpp` and `binary.cpp` - the clearest, smallest
   example of the full classify -> name -> dispatch pipeline.
5. `mlx/backend/metal/kernels/defines.h` - the `instantiate_kernel` macro.
6. `mlx/backend/metal/kernels.h` - `get_*_kernel` API and
   `get_template_definition` (the JIT-side mirror of `instantiate_kernel`).
7. `mlx/backend/metal/device.cpp` - library/pipeline loading and caching
   (`get_library`, `get_kernel`, `build_library_`).
8. `mlx/backend/metal/reduce.cpp` - the reduction strategy dispatcher; the
   single most relevant file for Module 5.
9. `mlx/backend/metal/kernels/reduction/reduce_row.h` - `threadgroup_reduce`,
   the simdgroup-first cross-thread reduction pattern.
10. `mlx/backend/metal/matmul.cpp` - gemv vs steel routing (skim only).

---

## The Best Reading Order

### Pass 1: See the shape of the whole system

Read:

- `refs/mlx/docs/src/index.rst` (or the top-level README) for the product pitch.
- `mlx/array.h`
- `mlx/primitives.h`

Goal: understand that MLX is graph-first, and that a `Primitive` is the unit
of "one operation, forward and backward, CPU and GPU."

### Pass 2: Find where laziness ends

Read:

- `mlx/transforms.cpp` (`eval`)
- `mlx/backend/metal/metal.cpp` (`make_task`)
- `mlx/scheduler.h` (just enough to see the enqueue call)

Goal: locate the exact seam between "graph building" and "GPU work happens
now," analogous to `mag_dispatch(...)` in Magnetron, but arrived at lazily
instead of eagerly.

### Pass 3: Learn the elementwise dispatch pattern (the easiest full example)

Read, in this order:

- `mlx/backend/common/binary.h` (`BinaryOpType` classification)
- `mlx/backend/metal/binary.cpp` (kernel naming, grid sizing, encoding)
- `mlx/backend/metal/kernels/binary_ops.h` (op functors: `Add`, `Multiply`, ...)
- `mlx/backend/metal/kernels/binary.h` (kernel templates: `binary_vv`,
  `binary_g`, ...)
- `mlx/backend/metal/kernels/binary.metal` (the `instantiate_binary_all`
  macro - the AOT instantiation catalog)

Goal: trace one op (say, `Add`) from `eval_gpu` all the way to a concrete
`[[host_name("vv_Addfloat32")]]` Metal function. This is the template for
every other op family in the codebase.

See [`MLX_KERNELS.md`](MLX_KERNELS.md) for the full annotated version of this
pass, including JIT vs AOT and the exact macros.

### Pass 4: Learn reduction (most relevant to current course module)

Read:

- `mlx/backend/common/reduce.h` and `reduce_utils.cpp` (`ReductionPlan`,
  `get_reduction_plan`)
- `mlx/backend/metal/reduce.cpp` (`all_reduce_dispatch`,
  `row_reduce_general_dispatch`, `strided_reduce_general_dispatch`)
- `mlx/backend/metal/kernels/reduction/ops.h` (generic `Op` structs: `Sum`,
  `Prod`, `Max`, `Min`, and `simd_reduce`)
- `mlx/backend/metal/kernels/reduction/reduce_all.h` (whole-array reduction)
- `mlx/backend/metal/kernels/reduction/reduce_row.h` (`threadgroup_reduce`,
  the core cross-thread pattern)
- `mlx/backend/metal/kernels/reduction/reduce_col.h` (strided/column
  reduction, for contrast)

Goal: understand the three-way classification (all / row / column), the
simdgroup-first reduction pattern, and the "never use atomics, always
multi-pass" rule for large reductions.

See [`MLX_REDUCTION.md`](MLX_REDUCTION.md) for the deep, annotated version of
this pass, written specifically against comtam's own `reduction.metal` WIP.

### Pass 5: Skim matmul for orientation only

Read:

- `mlx/backend/metal/matmul.cpp` (just the gemv-vs-steel routing logic)
- Directory listing of `mlx/backend/metal/kernels/steel/`
- One short excerpt of `steel/gemm/gemm.h`'s main K-loop

Goal: know that steel exists and roughly why, without trying to absorb the
whole GEMM library. This is explicitly a "later" reference.

### Pass 6: JIT vs AOT, if you want the full build-system picture

Read:

- `CMakeLists.txt` (root) - the `MLX_METAL_JIT` option
- `mlx/backend/metal/CMakeLists.txt` - `make_jit_source`, source selection
- `mlx/backend/metal/kernels/CMakeLists.txt` - `build_kernel`, the AOT
  `.air` -> `mlx.metallib` pipeline
- `mlx/backend/metal/jit_kernels.cpp` vs `nojit_kernels.cpp` - same function
  signatures, two implementations
- `mlx/backend/metal/device.cpp` - `get_library`, `get_kernel`,
  `build_library_`

Goal: understand that JIT/AOT is a build-time switch producing two
implementations behind the identical `kernels.h` API, not a runtime decision.

---

## Questions To Ask While Reading

1. Why does MLX bother with three levels of indirection (template `.h` ->
   instantiation `.metal`/JIT-string -> `[[host_name]]` lookup) instead of
   hand-writing every kernel variant? What would break if it hand-wrote them?
2. Why is layout classification (contiguous / broadcast / general) done in
   C++ on the host, never in the Metal kernel itself?
3. Why does MLX prefer a second kernel launch with an intermediate buffer
   over atomics for large reductions? What would an atomic-based version cost
   in practice on Apple GPUs?
4. Why is `simd_reduce` (32-lane hardware reduction) preferred over a
   threadgroup-memory halving tree, and when would the tree still be
   necessary (hint: look at the 8-byte-type branch of `DEFINE_SIMD_REDUCE`)?
5. What is the actual trigger condition for routing matmul to `gemv` instead
   of `steel`, and why is that condition ("one dimension is 1") a clean
   dividing line?

---

## Final Takeaway

MLX is not "what comtam should look like eventually" in a literal sense -
comtam is deliberately eager, single-device, and float32-first, and should
stay that way per the course thesis. MLX is useful as:

- an oracle for forward correctness (`docs/note/MLX_C_ORACLE.md`),
- a demonstration of where kernel-organization complexity *actually* comes
  from once an op has more than one dtype and more than one layout to handle,
  and
- a library of well-tested small patterns - simdgroup-first reduction,
  layout classification before dispatch, `[[host_name]]`-keyed template
  instantiation - that are worth borrowing individually, long after they are
  worth borrowing wholesale.

Read Magnetron to learn how to structure a runtime. Read MLX to learn how a
handful of Metal kernel patterns scale from "one op, one dtype" to "every op,
every dtype, every layout" without becoming unmaintainable.
