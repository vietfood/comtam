# Luminal Metal Backend Study Guide

**Upstream:** [luminal-ai/luminal](https://github.com/luminal-ai/luminal). Every path below is relative to that repository's root.

These are distilled reading notes on `crates/luminal_metal/`, the Metal backend of the Luminal inference compiler. It is the Metal-specific companion to [`LUMINAL.md`](LUMINAL.md), in the same role as [`MLX_KERNELS.md`](MLX_KERNELS.md) is to [`MLX.md`](MLX.md).

Why this crate earns a study guide: it is the only reference in these notes that is a **complete, tested, production Metal backend small enough to read end to end** (~4k lines of Rust in `src/`). MLX's Metal backend is far richer but embedded in a much larger C++ codebase; Magnetron has no Metal backend at all. The API patterns - device, buffers, shader compilation, pipeline caching, MPS, command encoding - translate almost 1:1 to metal-cpp, because both are thin wrappers over the same Objective-C objects.

One caveat to hold throughout: this is a **graph** backend. Luminal compiles an LLIR graph first, then executes it, so several designs below (per-node pipelines, intermediate-buffer planning, dim buckets) answer questions an eager runtime never asks. Each section says what transfers to comtam and what does not.

---

## File Map

| File | Lines | Role |
| --- | ---: | --- |
| `src/lib.rs` | 14 | crate root; re-exports `MetalRuntime`, `MetalOps` |
| `src/runtime.rs` | 851 | device/queue, pipeline cache, buffer maps, `execute()` |
| `src/memory_analysis.rs` | 1478 | compile-time peak/live memory estimation for search |
| `src/dyn_backend.rs` | 82 | dynamic-backend wrapper |
| `src/kernel/mod.rs` | 221 | `MetalKernelOp` trait, `MpsKernelCache`, encode context |
| `src/kernel/ops.rs` | 3468 | all kernels: MSL codegen, MPS + generic matmul, data ops |
| `src/kernel/matmul.rs` | 5 | just a layout enum - no shaders here |
| `src/tests.rs` | 1626 | correctness tests against Candle CPU + closed-form oracles |

The `MetalOps` tuple (`ops.rs:28`-`55`) is the complete list of Metal ops the backend can execute - the backend-side mirror of the 15 primitives from [`LUMINAL.md`](LUMINAL.md) section 1, plus matmul variants.

---

## 1. Shader compilation: MSL source strings, compiled at runtime

Luminal does not ship a `.metallib`. Every kernel is an MSL source string assembled in Rust and compiled on demand (`crates/luminal_metal/src/kernel/ops.rs:57`-`73`):

```57:69:crates/luminal_metal/src/kernel/ops.rs
fn compile_shader(device: &Device, source: &str, function_name: &str) -> ComputePipelineState {
    let options = metal::CompileOptions::new();
    options.set_language_version(MTLLanguageVersion::V2_4);
    let library = device
        .new_library_with_source(source, &options)
        .unwrap_or_else(|err| {
            panic!("Failed to compile Metal shader {function_name}: {err:?}\n{source}")
        });
    // ... get_function -> new_compute_pipeline_state_with_function
}
```

Note the error path: the full generated source is printed on failure, which is the only sane way to debug *generated* shaders.

**The honest tradeoff against comtam:** comtam deliberately moved the other way - `core::kernel_library` loads a build-time `default.metallib`, and `ARCHITECTURE.md` calls runtime source compilation "the training-wheels version." Both are right for their context:

- Luminal *generates* MSL per graph node (index math and dtypes are baked into the source, section 2), so it cannot precompile - the source does not exist at build time. Runtime compilation is a necessity, and the pipeline cache (section 3) makes it a once-per-graph cost.
- comtam's kernels are a fixed, hand-written set under version control, so AOT compilation catches shader syntax errors at build time and keeps startup deterministic.

The lesson is not "runtime compilation is sloppy"; it is "match the compilation timing to when the source becomes known." If comtam ever generates kernels (fusion experiments, shape-specialized variants), this crate is the reference for doing it cleanly.

## 2. Kernel codegen: specialize the source, not the pipeline count

Dtypes are handled by **string substitution**, not Metal-side templates (`ops.rs:90`-`115`):

```text
metal_buffer_type:  DType::F32 -> "float", F16 -> "half", Int -> "int"
metal_numeric_read: F16/Int values read through float(...)
metal_numeric_write: cast back on store
```

Index math for the specific input/output strides is lowered directly into the MSL body, so the compiled kernel contains plain integer arithmetic with no runtime stride arrays for the common cases. A representative elementwise binary kernel (`MetalAdd`, `ops.rs:469`-`494`) has the skeleton:

```text
kernel void mkernel(
    device A *a [[buffer(0)]],
    device B *b [[buffer(1)]],
    device O *out [[buffer(2)]],
    constant int *dyn [[buffer(3)]],
    constant uint &n_elements [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < n_elements) {
        out[...] = (a[...]) + (b[...]);   // index math baked in
    }
}
```

The reduction kernel (`MetalSumReduce`, `ops.rs:1034`-`1084`) is a classic threadgroup tree reduce: one threadgroup per output element, 256 threads, shared `partials[]`, barriers - worth comparing against comtam's own reduction kernels, and against MLX's more elaborate reduction machinery in [`MLX_REDUCTION.md`](MLX_REDUCTION.md).

This is a third point on the "how to multiply kernels by dtypes" curve (compare MAGNETRON.md section 9 and MLX_KERNELS.md): magnetron multiplies compiled C translation units per ISA, MLX instantiates C++ templates into named Metal functions, Luminal generates the whole MSL string per (op, dtype, layout). All three keep the *host-visible* kernel lookup stable and vary the bytes behind it.

## 3. Pipeline caching: keyed by graph node, not by name

```35:35:crates/luminal_metal/src/runtime.rs
pipelines: FxHashMap<NodeIndex, ComputePipelineState>,
```

Every kernel op implements one compile entry point (`kernel/mod.rs:146`-`152`):

```146:152:crates/luminal_metal/src/kernel/mod.rs
pub trait MetalKernelOp: EgglogOp {
    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<ComputePipelineState>;
}
```

filled once when the graph is loaded (`runtime.rs:658`-`662`). Returning `None` is meaningful: MPS matmul ops skip `ComputePipelineState` entirely and override `encode` to talk to Metal Performance Shaders (section 5), and the scatter op uses the same hook to stash a second copy kernel in a `OnceLock` (`ops.rs:2776`).

Contrast with comtam: `core::kernel_library` caches pipelines **by kernel name string** (`"add_fp32"`), because eager dispatch knows only the op, not a graph node. Both are "compile once, then hit a map"; the key differs because the execution model differs. comtam's by-name cache is the right eager design - but note how Luminal's per-node key lets two nodes running the "same" op hold differently-specialized pipelines, something a name-keyed cache cannot express without growing a richer key.

## 4. Buffers: shared storage everywhere, grow-only intermediates

Every allocation in the crate uses `MTLResourceOptions::StorageModeShared` (inputs `runtime.rs:118`-`123` and `516`-`540`, the dynamic-dim table `339`-`342`, intermediates `582`-`585`). There is no `StorageModePrivate`, no staging, and - notably - **no `MTLBlitCommandEncoder` anywhere**: even scatter's "copy destination to output first" step is a compute `copy_kernel` on the same encoder (`ops.rs:2906`-`3027`).

Intermediate buffers (one per non-aliasing kernel node) follow a grow-only policy (`runtime.rs:551`-`587`): allocated at graph-load time, reallocated only if a later request is bigger, never freed per-execute, cleared on graph switch. This validates comtam's current choice (shared buffers, synchronous copies) as a legitimate production posture for an Apple-silicon-first runtime, not merely a beginner's shortcut - unified memory means the shared/private split is a performance question, not a correctness one, and Luminal defers it entirely.

`memory_analysis.rs` (1.5k lines) is the one part that does **not** transfer: it is an egglog pass that computes live/peak intermediate memory per candidate graph so the *search* can rank memory alongside speed. An eager runtime has no candidates to rank. File the idea away for the execution and memory-lifetime work as evidence that "how much memory does this op sequence need" is a real question with real tooling - but the mechanism belongs to a compiler.

## 5. Matmul: MPS first, a hand-written kernel for the layouts MPS won't take

Two implementations coexist, and the compiler picks between them:

- **`MPSMatmul` / `MPSBatchedMatmul`** (`ops.rs:1542`-`1637`, batched `1925`-`2032`): `compile` returns `None`; `encode` builds `MPSMatrix` descriptors and a cached `MPSMatrixMultiplication`, encoding straight onto the command buffer. MPS objects are cached in `MpsKernelCache` (`kernel/mod.rs:39`-`122`) keyed by concrete `(rows, cols, row_bytes, dtype)` and `(m, n, k, transpose, alpha, beta)` - because MPS kernels are configured per shape, the cache key must include the shape.
- **`GenericMatmul`** (`ops.rs:2202`-`2345`): a fused multiply-accumulate compute shader with threadgroup reduction, used when layouts don't fit the MPS patterns (e.g. the non-contiguous merged-head projection covered by the test at `tests.rs:871`).

The e-graph rewrites that recognize "this `Mul`+`SumReduce` pattern is a matmul" live at `ops.rs:1327`-`1421`, keyed on the layout enum from `kernel/matmul.rs` (all 5 lines of it - row-major vs transposed-row-major).

The comtam mapping is direct, and it belongs to the performance track: keep the naive strided matmul as the correctness reference, add an MPS path for contiguous float32 GEMM shapes, and select per call site by layout. Luminal shows both halves of that pattern - and shows that "MPS object cache keyed by shape" is the part that is easy to get wrong.

## 6. Dynamic dimensions: a 26-slot side buffer

```18:18:crates/luminal_metal/src/kernel/mod.rs
pub const DYN_SLOT_COUNT: usize = 26;
```

Symbolic dims `'a'..'z'` resolve through one shared `int[26]` buffer (`runtime.rs:339`-`342`), written by the host before encoding (`runtime.rs:768`-`782`) and read in MSL as `dyn[i]` after a textual substitution of `const_a`...`const_y` (`ops.rs:75`-`84`). It is bound as the last buffer argument on every kernel (`kernel/mod.rs:181`-`185`). Optional "dim buckets" go further and keep multiple compiled graph variants for different size ranges (`runtime.rs:736`-`766`).

Transfer value for comtam: **none today**, and that is the point of listing it. Eager execution always knows concrete sizes, so sizes go in ordinary kernel arguments (`view_desc`/`kernel_desc`), where they are type-checked and obvious. Read this section as the price laziness pays for shape generality - a side channel that bypasses the type system and must be kept in sync by hand.

## 7. Command encoding: one buffer, one encoder per op, synchronous wait

Per `execute()` (`runtime.rs:409`-`455`):

```text
one MTLCommandBuffer
  for each op in the graph:
    new compute encoder -> bind pipeline + buffers + dyn -> dispatch -> end_encoding
    (MPS ops encode onto the same buffer without a compute encoder)
commit()
wait_until_completed()
```

Fully synchronous, exactly like comtam's current `metal_device` submit paths (`commit` + `waitUntilCompleted`, error-checked). Two differences worth knowing about when async execution arrives:

- Luminal pays only **one** commit+wait per *graph*, while comtam pays one per *op*. That is the structural overhead eager execution carries, and quantifying it is a measurement question.
- The timed path (`runtime.rs:830`-`848`) reads `GPUStartTime`/`GPUEndTime` off the completed command buffer - the simplest possible GPU-side profiling hook, directly stealable when comtam needs its first real measurement.

## 8. Tests: external oracle, realistic shapes

`tests.rs` (1.6k lines) validates the backend against:

- **Candle on CPU** for composite blocks - matmul, RMSNorm, attention, SwiGLU, a mini-transformer (`tests.rs:684`+, reference helpers at `181`+),
- closed-form host math and proptest for primitives (`369`+),
- exact-integer checks where tolerances would hide bugs (`348`+).

The flow is always: build graph -> compile/search -> `set_data` -> execute -> `get_f32` -> `assert_close` with a stated relative tolerance. Same discipline as comtam's mlx-c oracle tests: an independent implementation, small deterministic shapes, explicit tolerances.

---

## Pattern Mapping To comtam

| luminal_metal pattern | comtam equivalent today | When to revisit |
| --- | --- | --- |
| Runtime MSL source compilation | build-time `default.metallib` | only if kernels become *generated* (fusion, shape specialization) |
| Dtype by string substitution | `COMTAM_DISPATCH_DTYPE` + `[[host_name]]` names | when a second dtype is earned |
| Pipeline cache keyed by graph node | `kernel_library` cache keyed by name | if per-call-site specialization is ever measured to matter |
| `StorageModeShared` for everything | `core::storage` shared buffers | Only with a measurement showing shared is the bottleneck |
| Grow-only intermediate reuse | allocate per op result | Allocator reuse, after lifetime tests exist |
| MPS matmul + generic fallback | naive `matmul.metal` | MPS for contiguous GEMM, keep naive for strided views |
| One commit+wait per graph | one per op | Measure per-op commit overhead first |
| `GPUStartTime`/`GPUEndTime` timing | none | Steal this hook for the first GPU-side measurements |
| `dyn[26]` symbolic-dim buffer | concrete sizes in `view_desc` | never, unless comtam goes lazy (out of scope) |
| `memory_analysis.rs` (search-time memory ranking) | none | A question to answer, not code to port |

## Reading Order

1. `src/kernel/mod.rs` (221 lines, read fully) - the trait, the encode context, the MPS cache. The whole backend in miniature.
2. `compile_shader` and one elementwise kernel in `src/kernel/ops.rs` (`57`-`115`, `469`-`494`) - codegen end to end.
3. `MetalSumReduce` (`ops.rs:1034`-`1084`) - compare with comtam's reduction and MLX's (`MLX_REDUCTION.md`).
4. `src/runtime.rs` `execute()` (`409`-`455`) plus buffer allocation (`551`-`614`) - the execution core.
5. MPS matmul encode (`ops.rs:1565`-`1637`) and `GenericMatmul` (`ops.rs:2202`-`2345`) - the two-speed matmul strategy.
6. Skim `src/tests.rs` - how a backend stays honest.
7. `src/memory_analysis.rs` - skim only, to recognize compiler machinery when you see it.
