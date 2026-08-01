# Luminal Study Guide

This note is a map for studying the `refs/luminal/` subtree inside this
repository. It complements [`MAGENETRON.md`](MAGENETRON.md) and
[`MLX.md`](MLX.md): Magnetron is the small eager-runtime reference whose arc
comtam borrows; MLX is the production Apple GPU kernel reference; Luminal is
the **compiler-first counterpoint** - a rigorous, small implementation of the
exact architecture comtam deliberately rejects, plus two ideas comtam does
adopt:

1. **The RISC-style primitive op set** - proof that ~15 primitives generate
   transformers, convnets, and nearly everything else. comtam's mandatory
   Module 5A semantic-surface audit is the eager version of this idea.
2. **A production Metal backend** (`crates/luminal_metal`) - shader-from-source
   compilation, pipeline caching, MPS matmul, and graph-level buffer reuse.
   Covered in the companion note [`LUMINAL_METAL.md`](LUMINAL_METAL.md).

Read Luminal to sharpen the eager/lazy boundary and to mine the primitive op
set - not as an architecture template.

---

## What Luminal Is

Luminal is a high-performance inference compiler written in Rust. Its
distinguishing choices are the opposite of comtam's course thesis:

1. **Lazy, compile-everything execution.** Writing `a.matmul(b)` records nodes
   into a static computation graph; nothing runs until `compile()` +
   `execute()`. There is no eager mode at all.
2. **A small primop set.** Everything lowers to 15 primitive ops. High-level
   ops are frontend compositions over them.
3. **Search instead of heuristics.** Compilation is an e-graph (egglog)
   exploration of semantically-equivalent graphs, ranked by actually profiling
   candidates on the target device. This is how Luminal "discovers" fusions
   like Flash Attention instead of hand-writing them.
4. **First-class dynamism via symbolic shapes.** Dimensions are `Expression`s
   (variables + arithmetic), so one compiled graph serves many shapes.

The consequence to keep in mind while reading: Luminal can afford aggressive
decomposition because its compiler fuses the pieces back before execution.
Nothing in comtam performs that rescue, so the decomposition strategy itself
does not transfer - the *op inventory* does.

---

## Architecture In One Diagram

```text
GraphTensor API (frontend/)
  matmul, softmax, mean, sub, div, ... written as compositions
        |
        v
HLIR graph (hlir.rs + graph.rs)
  15 primitive ops + Input/Output/Constant/Loop nodes
  petgraph DAG, ShapeTracker metadata on every edge
        |
        v
egglog e-graph compilation (graph.rs, egglog_utils/)
  equivalence rewrites + search over candidate graphs
  candidates profiled on the real device, best wins
        |
        v
Runtime (trait Runtime, op.rs)
  load_llir / execute / profile
        |
        +--> luminal_metal     (Metal buffers, MPS, source-compiled kernels)
        +--> luminal_cuda_lite
        +--> ReferenceRuntime  (CPU interpreter, the correctness oracle)
        |
        v
crates/luminal_training   autograd as a graph-to-graph transformation
crates/luminal_nn         nn modules written against GraphTensor
```

The important point for comtam readers: the *frontend* and the *primitive
inventory* are small and readable. Everything between them and the device
(e-graph, search, symbolic-dim specialization) is the complexity the small
op set is designed to control - and the part comtam skips by running eagerly.

---

## The Core Ideas

### 1. The 15 primitive ops are the whole contract

Every compute op in the framework is one of these structs in
`refs/luminal/src/hlir.rs`:

| Op | Defined | Semantics |
| --- | --- | --- |
| `Log2` | `hlir.rs:1332` | elementwise log base 2 |
| `Exp2` | `hlir.rs:1404` | elementwise 2^x |
| `Sin` | `hlir.rs:1476` | elementwise sin |
| `Recip` | `hlir.rs:1549` | elementwise 1/x |
| `Sqrt` | `hlir.rs:1622` | elementwise square root |
| `Add` | `hlir.rs:1755` | elementwise binary add |
| `Mul` | `hlir.rs:1855` | elementwise binary multiply |
| `Mod` | `hlir.rs:1955` | elementwise remainder |
| `LessThan` | `hlir.rs:2053` | elementwise `<`, producing `Bool` (not a float mask) |
| `SumReduce` | `hlir.rs:2468` | reduce one axis by summation |
| `MaxReduce` | `hlir.rs:2617` | reduce one axis by maximum |
| `Iota` | `hlir.rs:1100` | range/arange generation along an axis |
| `Gather` | `hlir.rs:2154` | index-select |
| `Scatter` | `hlir.rs:2316` | index-write |
| `Cast` | `hlir.rs:1161` | dtype conversion |

Plus structural (non-compute) nodes: `Input`, `Output`, `Constant`, and the
`Loop*` family (`hlir.rs:269`-`1061` region) used for rolled-loop execution.

Three properties make this set interesting rather than arbitrary:

- **Base-2 transcendental choice.** `Log2`/`Exp2` are the primitives; `exp`
  and `ln` are compositions (`unary.rs:100-107`: `exp(x) = exp2(x * 1/ln2)`,
  `ln(x) = log2(x) * ln2`). Base-2 matches what GPU fast-math units
  (`ex2.approx`, `lg2.approx` on CUDA; `exp2`/`log2` in MSL) compute directly.
- **One comparison op.** `LessThan` yields a `Bool` tensor; `Cast` turns it
  back into a numeric mask, and arithmetic over those masks generates every
  other comparison plus `maximum` (`binary.rs:296`-`388`; `maximum` itself is
  `(lt.cast * rhs) + (le.cast * lhs)` at `binary.rs:358`-`359`), which in turn
  generates `relu`, `clip`, `abs` (`unary.rs:266`-`278`).
- **Reductions are axis-local.** `SumReduce`/`MaxReduce` reduce one axis; a
  full reduce is a composition over axes, and `mean` is `sum / count`.

comtam's Module 5A baseline is this inventory filtered through eager
constraints: `add`, `mul`, `neg`, `recip`, `sum`, and `matmul` as
differentiable semantic primitives; `sub`, `div`, and `mean` are composed.
Physical dtype/layout kernel variants and internal backward-support kernels are
counted separately from that semantic surface. `exp2`/`log2`/`sqrt`/`sin` are
the documented future unary family (Module 17 territory), and `max_reduce` is
deferred until an operation earns it.

### 2. The frontend shows composition done honestly

`refs/luminal/src/frontend/` is the most directly transferable reading in the
repo: high-level ops written as plain Rust over the primitives, each one a
few lines. Representative examples:

```text
sub(a,b)   = a + (b * -1)                          binary.rs:51, unary.rs:69
div(a,b)   = a * recip(b)                          binary.rs:119
exp(x)     = exp2(x * 1/ln2)                       unary.rs:100
ln(x)      = log2(x) * ln2                         unary.rs:105
cos(x)     = sin(pi/2 - x)                         unary.rs:134
square(x)  = x * x                                 unary.rs:139
abs(x)     = relu(x) + relu(-x)                    unary.rs:266
sigmoid(x) = recip(1 + exp(-x))                    unary.rs:281
tanh(x)    = 2*sigmoid(2x) - 1                     unary.rs:297
softmax    = exp(x - max) / sum(exp(x - max))      unary.rs:193
```

And the cautionary one - matmul (`matmul.rs:4`-`118`) is broadcast-`mul` plus
`sum_reduce`:

```18:33:refs/luminal/src/frontend/matmul.rs
if (self.shape.len() == 1 || self.shape.len() == 2) && rhs.shape.len() == 2 {
    let vec = self.shape.len() == 1;
    if vec {
        self = self.expand_dim(0, 1);
    }
    let (m, _) = self.dims2();
    let (_, n) = rhs.dims2();
    // Broadcasted Multiply
    let mul = self.expand_dim(1, n) * rhs.permute((1, 0)).expand_dim(0, m);

    // Sum Reduce
    let mut ret = mul.sum(2);
    if vec {
        ret.shape.remove_dim(0);
    }
    ret
}
```

The `(m, n, k)` intermediate exists only as a graph pattern; the compiler
rewrites it into a real matmul kernel before execution. An eager runtime
evaluating this composition literally would allocate and traffic `m*n*k`
elements. This single function is the complete argument for why comtam's
Module 5A baseline keeps `matmul` as a primitive while composing
`sub`/`div`/`mean`.

### 3. ShapeTracker: a third stride-tracking design to compare

`refs/luminal/src/shape/tracker.rs` holds per-tensor metadata:

```12:18:refs/luminal/src/shape/tracker.rs
pub struct ShapeTracker {
    pub dims: ArrayVec<[Expression; 10]>,
    pub strides: ArrayVec<[Expression; 10]>,
    /// Bits per element in memory storage. Controls byte-size computation.
    /// Defaults to 32 (F32). Set from dtype.bits() at tensor creation.
    pub element_stride_bits: usize,
}
```

Worth comparing against magnetron's `mag_coords_t` (fixed `[16]` arrays,
concrete `int64_t`) and comtam's `view`:

- Same core idea: dims + strides, fixed-capacity inline storage, no per-tensor
  heap allocation for shape data.
- Movement ops are pure stride manipulation: `expand` inserts stride-0 axes
  (`tracker.rs:107`-`149`), `permute` reorders (`tracker.rs:187`-`197`).
- Notably, there is **no `reshape` API**. Reshaping is `merge_dims` /
  `split_dims` / `flatten` (`tracker.rs:338`-`399`), and `split_dims` simply
  panics when the old strides cannot separate cleanly - no silent copy. When a
  layout genuinely cannot be a view (a slice with a non-zero start, an output
  readback), the framework materializes through `Iota` + `Gather` instead
  (`movement.rs:504`-`533`). Contrast with comtam's Module 2 `reshape`, which
  rejects non-contiguous layouts for the same reason but with a dedicated op.
- Compute ops hand their result a fresh contiguous tracker
  (`shape.contiguous()`), so the *next* op always sees a row-major buffer even
  when its input was a view.
- The `Expression` layer (`shape/expression.rs`, 1.5k lines) makes dims
  symbolic (`(s, 4096)`, `(b, h, w+3)`) so one compiled graph covers many
  shapes. **This is the part comtam does not need** - eager execution always
  knows concrete sizes. Read it to understand what laziness forces a
  framework to pay for, not to import it.

### 4. Graph recording and compilers - read for contrast only

When you write `a.matmul(b)` on `GraphTensor`s, the frontend calls
`graph().add_op(...)` (e.g. `binary.rs:87`-`94`) which appends a typed node to
a petgraph DAG and returns a new `GraphTensor` handle. No kernel launches, no
allocation. Compilation then rewrites this DAG - via egglog e-graph rules
plus device-profiling-based search (`graph.rs`, 5.8k lines) - into a faster
equivalent graph, and a `Runtime` (`op.rs:60`) executes the result.

This machinery is the bulk of the repo and the least transferable part of it.
comtam's scope rules exclude lazy graphs, scheduling, and fusion; read
`graph.rs` only if you want a concrete answer to "what would eager comtam have
to grow to get fusion?" (Roughly: a DAG IR, symbolic shapes, an equivalence
engine, a profiler-in-the-loop search, and a runtime boundary - i.e., a
different framework.)

### 5. Autograd is a graph transformation over the primitive set

`crates/luminal_training` implements training as a graph-to-graph
transformation performed **before compilation**: `backward` walks the HLIR
graph in reverse topological order and, for each node, pattern-matches its op
and appends ordinary VJP nodes into the *same* graph (`autograd.rs` in that
crate) - `Mul` emits `g*other`, `Exp2` emits `g*out*ln2`, `SumReduce` emits an
expand, `Gather` emits a scatter-add. The single e-graph then optimizes the
forward and backward passes together.

The size data point is the argument: `luminal_training` is ~2.7k lines
(including tests) against a ~23k-line core, of which `graph.rs` +
`egglog_utils/` alone are ~11k. Because only 15 node types exist, "derive the
local gradient of every possible op" is a weekend-scale pattern match, not a
research project. That is exactly the economy comtam's Module 6 collects
dynamically - the tape records composed ops as their constituent primitive
nodes, so hand-written rules are needed only per primitive. Compare the two
once Module 6 is underway: statically-derived VJPs over a closed op set
(`luminal_training`) vs dynamically-recorded tapes (magnetron, comtam).

### 6. Correctness culture: a CPU interpreter plus an external oracle

Luminal tests on two levels, and the split maps directly onto comtam's oracle
discipline (Module 4):

- **`ReferenceRuntime`** (`hlir.rs:2920`-`3021`) is a CPU interpreter over the
  extracted primitive graph - the framework's own semantic ground truth. GPU
  backends can always be checked against "the primitives, executed plainly."
- **Candle** (an external Rust tensor library) is the *independent* oracle in
  both core tests (`src/tests/mod.rs`, shared harness at
  `frontend/binary.rs:433`-`520`) and Metal backend tests, plus proptest for
  randomized shape coverage. The external dependency is what keeps the
  comparison honest - the same reason comtam tests against mlx-c rather than
  its own CPU reimplementation.

### 7. The Metal backend is the other reason this ref exists

`crates/luminal_metal` is a complete, tested Metal backend: MSL source
strings compiled at runtime, per-kernel `ComputePipelineState` caching,
MPS-backed matmul, and a graph-level buffer-reuse analysis. See the
companion note [`LUMINAL_METAL.md`](LUMINAL_METAL.md) for the deep dive and
the mapping onto comtam's `core::kernel_library` / `metal_device` design.

---

## What To Borrow vs. What To Leave

Borrow:

- The primitive op inventory (section 1) as the reference for comtam's kernel
  budget decisions, now and when future modules earn new ops.
- The frontend composition patterns (section 2) when writing composed ops -
  they are small, correct, and already thought through.
- The base-2 choice for transcendental primitives when Module 17 adds
  `exp2`/`log2`.
- `ShapeTracker`'s fixed-capacity dims/strides as a comparison point for
  comtam's `view` (Module 2/9 redesign questions only).
- The dual-oracle testing posture (section 6): a CPU interpreter as internal
  ground truth plus an external library (Candle) as the independent check -
  the same discipline comtam applies with mlx-c.
- The Metal backend's concrete patterns - see `LUMINAL_METAL.md`.

Leave:

- The e-graph/egglog/search compiler stack (`graph.rs`, `egglog_utils/`).
- Symbolic `Expression` shapes (`shape/expression.rs`).
- The `Runtime` trait's compile/profile/execute lifecycle and `dyn_map`
  plumbing.
- Multi-backend structure (`luminal_cuda_lite`, `dyn_backend.rs`) and the
  dtype zoo (`dtype.rs`, FP8/FP4 paths).
- Rust idioms generally - the value here is architecture and Metal usage, not
  language technique.

---

## Reading Order

A few hours, in this order:

1. `refs/luminal/README.md` and `docs/docs/why.mdx` - the ideology, stated
   better than any secondary summary. Note where you disagree, given comtam's
   thesis.
2. `src/frontend/binary.rs` and `src/frontend/unary.rs` - read the
   compositions in section 2 directly; each is a few lines.
3. `src/frontend/matmul.rs` - the broadcast-mul + sum-reduce matmul. Have the
   eager-memory-cost argument ready before and after reading.
4. `src/hlir.rs` - skim the 15 op structs (line refs in section 1). Note how
   little semantic surface each one has.
5. `src/shape/tracker.rs` - skim movement ops; compare with comtam's `view`.
6. `src/graph.rs` - skim only: `add_op`, the `Graph` struct, and the compile
   entry points, to see the shape of what laziness costs.
7. [`LUMINAL_METAL.md`](LUMINAL_METAL.md) and then `crates/luminal_metal` -
   the second reason this ref exists.

---

## Questions To Ask While Reading

1. Which of the 15 primitives has the most surprising absence (no `Sub`, no
   `Div`, no `Matmul`, no `Exp`), and what composes it?
2. Why is `LessThan` enough to build `maximum`, `clip`, and `relu`? What does
   the gradient of `maximum` look like when built this way?
3. If comtam evaluated `matmul.rs`'s composition eagerly for a (512, 768) x
   (768, 512) multiply, how many elements would the intermediate hold versus
   the inputs?
4. What does Luminal's `Runtime` trait buy that comtam's "op runs now"
   deliberately gives up - and what does it cost in core-crate size (compare
   `src/graph.rs` against all of comtam)?
5. Which Luminal choices exist only to serve search/compilation (symbolic
   shapes, `Expression`, profiling hooks) and would be dead weight in an
   eager runtime?
