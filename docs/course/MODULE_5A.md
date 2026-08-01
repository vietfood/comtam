## Module 5A: Primitive Surface Consolidation

Turn the forward operations from Module 5 into a small, explicit semantic
surface before autograd depends on it. This is a required module, not optional
cleanup: Module 6 may start only after both the Module 5 and Module 5A gates
pass.

## Why This Is A Separate Module

Module 5 answers shape questions: broadcasting, reductions, and matrix
multiplication. This module answers a different architectural question:

```text
Which operations are independently dispatched and differentiated,
and which public operations are compositions of those primitives?
```

Changing that answer while building autograd would mix forward refactoring with
graph debugging. A separate gate keeps failures attributable: Module 5 proves
the shape-changing forward runtime, Module 5A consolidates its operator
contract, and Module 6 differentiates that tested contract.

## Module Contract

**Prerequisite:** Module 5 has passed with rank-0-through-rank-4 broadcasting,
full and axis reduction, strided-input matmul, unsupported-case validation, and
independent forward-oracle coverage.

**You will produce:**

- primitive unary `neg` and `recip` operations with a narrow unary dispatch path
- composed public `sub`, `div`, full `mean`, and axis `mean`
- shared public preflight validation that completes before allocation or dispatch
- an explicit distinction between semantic ops and physical Metal entry points
- documented floating-point behavior for negation, reciprocal, and division
- a clean removal of dispatchable `sub` and `div` kernels and `Op` entries
- an explicit forward-only policy for implemented primitives whose autograd
  semantics are intentionally deferred
- a tape-boundary decision that Module 6 can implement without duplicate
  broadcast nodes

**Supported scope:** float32, one explicit `core::context`, rank 0 through rank
4, positive-sized dimensions, right-aligned broadcasting, non-negative axes,
and the Module 5 matmul scope. Rank above four and zero extents reject before
allocation or dispatch. Module 9 may broaden empty-tensor behavior later.

**Not required:** fusion, a generic expression system, scalar-kernel overloads,
constant caches, higher-rank GPU metadata, in-place arithmetic, operator
overloading, or performance promotion based only on intuition.

**Completion evidence:** a design inventory, independent forward-oracle tests,
validation tests, documented numerical cases and tolerances, a clean build, and
the exact commands recorded in `docs/solution/MODULE_5A.md`.

## Four Different Surfaces

Do not use “kernel,” “primitive,” and “operation” as synonyms. This module fixes
four related but different surfaces:

| Surface | Module 5A contents | Contract |
| --- | --- | --- |
| Public tensor API | `add`, `sub`, `mul`, `div`, `neg`, `recip`, `sum`, `mean`, `max`, `min`, `matmul`, movement ops | What callers can request |
| Differentiable semantic primitives | `add`, `mul`, `neg`, `recip`, `sum`, `matmul`, movement ops | Nodes that Module 6 records and gives local rules |
| Forward-only semantic primitives | `max` | Independently dispatched and tested; Module 6 must reject gradient recording until a tie policy and backward rule are added |
| Composed public ops | `sub`, `div`, `mean`, `min` | Ordinary tensor-layer code; no dedicated node or rule; `min` inherits the forward-only autograd policy from `max` |
| Physical Metal entry points | dtype/layout variants of the dispatched primitives | Backend implementation detail, not the semantic count |

At the Module 5A gate, the independently dispatched arithmetic families are:

```text
binary:     add, mul
unary:      neg, recip
reduction:  sum_reduce
forward-only reduction: max_reduce
matrix:     matmul
```

This is six initially differentiable arithmetic primitives plus one
forward-only semantic primitive, but not necessarily seven Metal function
names. A contiguous and a strided matmul entry point are two physical
kernels implementing one semantic `matmul` operation. Conversely, Module 6 will
add an internal backward-support kernel for zero-padding without making
`zero_pad` a public differentiable operation.

The inventory is deliberately scoped in time. Module 7 earns `relu`, and Module
8 has a dedicated softmax-cross-entropy path. “Small” means every addition has a
specific semantic or runtime reason; it does not mean the list can never grow.

## The Composition Decision

The public compositions are:

```text
sub(a, b)        = add(a, neg(b))
div(a, b)        = mul(a, recip(b))
mean(a)          = mul(sum(a), scalar(1.0F / a.numel()))
mean(a, axis, k) = mul(sum(a, axis, k),
                       scalar(1.0F / a.shape[axis]))
min(a, ...)      = neg(max(neg(a), ...))
```

`sub`, `div`, and `mean` keep their public names because they express useful
intent. They do not keep dedicated semantic nodes merely because they have
public names. When recording is enabled in Module 6, callers get the graph
formed by the primitive calls above. The mean scale is a positive finite rank-0
constant in the supported scope and does not require gradients, so it must not
become a tape node.

### Why `neg` Is A Primitive

`neg(x)` could itself be written as `mul(x, -1)`, so a dedicated kernel is not
forced by asymptotic complexity. It is chosen here for three narrower reasons:

1. Negation has direct floating-point semantics, including signed zero, without
   constructing and uploading a scalar tensor.
2. It is reused by `sub` and by backward formulas, so the unary dispatch path is
   exercised by more than reciprocal alone.
3. When both operands require gradients, the tape for `sub(a, b)` becomes
   exactly `neg(b)` followed by `add(a, ...)`, with no framework-created
   constant tensor in the graph. Normal `requires_grad` propagation may omit
   the negation node when `b` is constant.

The cost is real: one more shader, forward oracle, local gradient rule, and
dispatch case. Record that tradeoff in the solution note. Do not claim that
`neg` makes the primitive count smaller.

This intentionally differs from Luminal, which composes negation from multiply
inside a lazy graph ([`../refs/LUMINAL.md`](../refs/LUMINAL.md)). Luminal can
fold constants and fuse the graph before execution; eager comtam pays each
constant upload and launch directly. Borrow the small-surface reasoning, not
the exact primitive list.

### Why `sub`, `div`, And `mean` Are Composed

Their compositions allocate only linear-sized intermediates and preserve the
public shape contract. This is acceptable for the correctness-first eager
runtime. Matmul is different: broadcast-multiply plus reduction would
materialize an `(m,n,k)` intermediate, changing memory from `O(mk + kn + mn)`
to `O(mnk)`, so it keeps a dedicated kernel.

Composition still has costs: extra launches, intermediate buffers, and
different rounding from a fused instruction. Measure those costs in Module 13
before promoting a composed operation to a dedicated implementation.

## Assignment 5A.1: Write The Operator Inventory ⭐⭐

**Task:** Before editing dispatch code, write a table in the solution note with
one row per public operation and these columns:

```text
public API | semantic primitive or composition | physical entry point(s)
validation owner | forward oracle | future backward owner
```

The table must distinguish operation families from dtype/layout variants. It
must also state that internal backward-support kernels are allowed later
without becoming public differentiable primitives.

**Questions:**

1. Why should the autograd rule registry follow semantic nodes rather than
   shader function names?
2. If Module 13 adds a fused `sub_fp32`, must Module 6 gain a `sub` rule? State
   which semantic graph that optimized path promises to preserve.
3. Which current `Op` values name dispatchable primitives, and which names must
   disappear after composition?

## Assignment 5A.2: Add Unary `neg` And `recip` ⭐⭐⭐

**Task:** Add a unary operation path with strided input views and contiguous
outputs. Implement `neg_fp32` and `recip_fp32` in `unary.metal`, add only the
primitive `Op` entries and name mappings, and choose either a narrow
`submit_uop` descriptor or a deliberately shared descriptor.

The unary kernel contract is:

```text
input:   one float32 tensor view, rank 0..4, positive numel
output:  a new contiguous tensor with the same logical shape
mapping: one output element per thread; input read through shape/stride/offset
```

Reject unsupported rank or extent before output allocation. Do not make unary
ops contiguous-only merely because their first tests use contiguous inputs.

**Tests:** Compare both operations with an independent oracle on a scalar,
ordinary contiguous shapes, a transpose, and a non-zero-offset shrink. Include
rank-5 and zero-extent rejection. Use ordinary finite values for accuracy tests;
the exceptional-value matrix belongs in Assignment 5A.5.

## Assignment 5A.3: Make Validation Precede Composition ⭐⭐⭐

**Task:** Extract pure or side-effect-free preflight helpers for unary, binary,
and reduction contracts. A public call must finish all validation that can be
known from metadata before it allocates an output, uploads a constant, or
submits an internal primitive.

For a binary operation, preflight returns at least the validated broadcast
shape after checking:

- both dtypes match and are supported
- both ranks fit the rank-4 GPU descriptor
- neither shape contains a zero or negative extent
- the shapes are broadcast-compatible
- every other public restriction documented by the operation

This ordering matters in an eager runtime. A naïve `sub(a, b)` that launches
`neg(b)` before discovering incompatible `a` and `b` has already performed GPU
work for an invalid public call.

**Required implementation boundary:** public binary semantic nodes refer to the
original operands. Stride-zero views created only to submit a binary kernel are
raw internal metadata aliases, not public `expand` operations and not future
GradNodes. Module 6 will therefore reduce `dout` to each original operand shape
exactly once in the binary rule. A user-authored `expand`, by contrast, is a
real public movement operation and records its own node.

**Tests:** Unit-test preflight helpers without Metal. Cover dtype mismatch,
rank 5, incompatible shapes, scalar broadcasting, and zero extent. Public API
tests must prove constructible invalid metadata reaches the same preflight for
primitive and composed binary ops. While float32 is the only constructible
dtype, helper-level dtype mismatch evidence is sufficient; do not manufacture
an invalid tensor solely for that test. Do not add profiling counters solely to
observe “no dispatch”; make the code structure enforce
preflight-before-execution.

## Assignment 5A.4: Compose `sub`, `div`, And `mean` ⭐⭐⭐

**Task:** Implement the public compositions after successful preflight:

```text
sub(a, b) = add(a, neg(b))
div(a, b) = mul(a, recip(b))
```

Implement both full and axis `mean` from the corresponding `sum` result and a
rank-0 scale tensor. Create that constant per call from the explicit context;
do not introduce a global or context cache before measurement justifies its
ownership and invalidation complexity.

Remove `Op::SUB` and `Op::DIV`, their name mappings, and their Metal kernels.
The public `tensor::sub` and `tensor::div` functions remain. Do not leave dead
dispatch cases “for later,” because their presence makes the semantic inventory
false.

Apply the same rule to unused speculative entries. `Op::MAX` is retained because
it is fully dispatched and has public forward-oracle coverage; it is a real
forward semantic primitive, not a placeholder. Its gradient is deliberately
deferred: Module 6 must reject `max` and composed `min` on a recording tensor
until a tie policy and gradient test are added.

**Tests:** Run the full public oracle matrix for `sub`, `div`, full `mean`, and
axis `mean`: scalar/tensor combinations, asymmetric ranks, non-contiguous
binary inputs, both `keepdim` values, invalid metadata, and positive-sized
boundary shapes. `mean` inherits the supported layouts of `sum`, which remains
contiguous-only at this gate. The tests observe public semantics, not whether
the implementation is a kernel or a composition.

## Assignment 5A.5: Define Numerical Semantics ⭐⭐⭐

**Task:** Replace the old division kernel's ad hoc zero-denominator behavior
with a documented float32 policy and test it. For this module, follow the
ordinary IEEE-style behavior produced by the Metal primitives and matched by
the independent oracle:

| Case | Required classification |
| --- | --- |
| finite nonzero / finite nonzero | finite result, checked with absolute and relative tolerance |
| nonzero finite or signed infinity / signed zero | signed infinity using operand-sign XOR |
| signed zero / signed zero | NaN |
| signed zero / finite nonzero | signed zero using operand-sign XOR |
| signed zero or nonzero finite / signed infinity | signed zero using operand-sign XOR |
| signed infinity / finite nonzero | signed infinity using operand-sign XOR |
| signed infinity / signed infinity | NaN |
| any arithmetic input NaN | NaN propagation |

Verify the Metal compiler settings used by the build. If an unsafe fast-math
mode invalidates these classifications, either disable it for the affected
kernels or explicitly narrow and test the supported public contract; do not
promise IEEE-style edges while compiling them away.

Do not compare NaN or infinity with an epsilon matcher. Assert classification
and sign explicitly where the platform contract provides it. Keep ordinary
reciprocal and division oracle inputs away from zero so one exceptional value
does not hide accuracy failures in the finite cases.

Test reciprocal directly as the `1/x` subset of the same policy:
`recip(+0) -> +inf`, `recip(-0) -> -inf`, signed infinity maps to the
corresponding signed zero, and NaN remains NaN.

Document that `mul(a, recip(b))` can round differently from one division
instruction. Choose one finite absolute-plus-relative tolerance for the public
division family from observed oracle evidence. Do not loosen it per shape or
per failing example.

For negation, explicitly test `neg(+0) -> -0`, `neg(-0) -> +0`, infinities, and
NaN classification. These cases are part of the reason `neg` was chosen as a
primitive instead of multiplication by a constant.

## Assignment 5A.6: Perform The Static And Clean-Build Audit ⭐⭐

**Task:** Prove that documentation, dispatch, shaders, and tests describe the
same semantic surface.

Required evidence:

- a clean build contains no `sub` or `div` Metal entry point
- `Op` contains only independently dispatched semantic operations, with no
  unwired future placeholder; retained `MAX` is documented as forward-only
- all GPU-facing rank checks reject above four before descriptor conversion
- public `sub`, `div`, and `mean` remain available and pass their oracle tests
- the current Module 5A inventory and architecture documentation identify
  primitives versus compositions; retroactive Module 4 grading text is not a
  gate when it has been superseded
- physical layout variants are listed without inflating the semantic op count
- exact build and focused/full CTest commands are recorded

## Module 5A Checklist

- [ ] 5A.1 Record the public, semantic, and physical operator inventories.
- [ ] 5A.2 Add strided unary `neg` and `recip` with independent oracles.
- [ ] 5A.3 Centralize metadata preflight and define the raw broadcast boundary.
- [ ] 5A.4 Compose `sub`, `div`, full `mean`, and axis `mean`; retire their redundant dispatch paths.
- [ ] 5A.5 Define and test finite and exceptional numerical semantics.
- [ ] 5A.6 Clean-build and audit code, tests, the current inventory, and architecture documentation.

## Exit Criteria

You are ready for Module 6 when:

1. Module 5 and Module 5A both pass independently; consolidation does not hide
   a missing shape-changing forward test.
2. `add`, `mul`, `neg`, `recip`, `sum`, `max`, and `matmul` have independent
   forward-oracle coverage on every supported layout relevant to them.
3. `sub`, `div`, full `mean`, and axis `mean` pass public oracle tests as
   compositions with no dedicated `Op` or semantic node.
4. Invalid public calls finish preflight before any internal primitive executes.
5. The documented rank-4 and numerical contracts have positive and negative
   tests, including NaN, infinity, signed zero, and zero denominator behavior.
6. Internal binary broadcast aliases are explicitly non-recording; user-authored
   movement operations remain visible to the future tape.
7. A build and full CTest run pass, and the solution note records exact evidence
   and remaining unsupported behavior. User-reported verification is acceptable
   when clearly labeled rather than presented as independently rerun evidence.
