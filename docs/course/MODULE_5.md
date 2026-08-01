## Module 5: Broadcasting, Reductions, And Matmul

Add the shape-changing forward operations that real models need. Keep the
existing public arithmetic surface working here; mandatory Module 5A will
consolidate its primitive/composed boundary before autograd begins.

## Module Contract

**Prerequisite:** Module 4 has independent forward-oracle coverage.
**Deliverables:** `broadcast_shapes`, broadcasted
float32 `add`/`sub`/`mul`/`div` using strided input views, full and axis `sum`,
full and axis `mean`, and 2-D `matmul` with a fast contiguous path plus a
strided-input path. Every new operation and materially different kernel path
requires public API correctness tests.

Supported scope is right-aligned NumPy-style broadcasting, rank-0 through
rank-4 tensors, positive-sized dimensions, non-negative axes, and 2-D matmul
only. Rank above four must reject before conversion to the fixed-size GPU view
descriptor. Zero-sized dimensions must reject before allocation or dispatch
because the current storage layer forbids zero-byte allocation; Module 9
revisits empty-tensor semantics deliberately. Negative axes, batched matmul,
dtype promotion, implicit materialization, and fusion are unsupported.

Use MLX-C on its CPU stream as the independent oracle for framework shape and
value semantics. A transparent manual CPU loop is optional when it makes a
failure easier to debug, but it is not duplicate gate work. Record which oracle
each test uses. Completion evidence is CTest coverage of the matrix below and
of every selected kernel path. Optimized kernels are valid learning work, but
they earn no performance claim without measurements.

| Area | Required evidence |
| --- | --- |
| Broadcast binary ops | all four public ops; compatible asymmetric ranks; scalar operand; non-contiguous input; incompatible shapes; rank-5 and zero-extent rejection |
| Reductions | full scalar result, each valid axis with `keepdim` both false and true, invalid axis, and zero-extent rejection |
| Matmul | ordinary 2-D case, either transposed input, bad rank/inner dimensions, and zero-extent rejection |

## Why This Module Exists

Module 3 gave you elementwise ops on identical shapes. Real networks need three
things that change shape or combine elements:

```text
broadcasting   add a (3,) bias to a (4, 3) batch
reductions     sum a (4, 5) down to a scalar or along an axis
matmul         the core of every linear layer
```

Each one stresses a different part of the runtime. Broadcasting stresses views.
Reductions stress kernels that coordinate across threads. Matmul stresses
layout. Get these right with oracles now, because Module 5A consolidates their
semantic surface and Module 6 builds gradients on top of it.

Module 5A deliberately comes next. It will decide which of these public
operations remain independently dispatched, define their numerical semantics,
and expose a tested primitive surface to Module 6. Do not perform that
consolidation opportunistically while reductions or broadcast correctness are
still failing.

## Part A: Broadcasting

### Mental Model

Broadcasting is alignment plus `expand`. You already have `expand` (stride 0)
from Module 2. Broadcasting is the rule for deciding which dimensions to expand.

The rule, aligning shapes from the right:

```text
(4, 3) and (3,)
  align:   (4, 3)
           (1, 3)
  result:  (4, 3)
```

A dimension is compatible if the sizes are equal or one of them is 1. The size-1
side expands with stride 0. No data moves.

### Assignment 5.1: Implement `broadcast_shapes` ⭐⭐

**Task:** A pure function that takes two shapes and returns the broadcasted
shape, or errors if they are incompatible.

Treat `()` as a scalar shape, compatible with every shape. Shape inference may
mathematically infer zero extents, but the Module 5 tensor operation must reject
them before allocation because empty storage is not supported yet. Keeping shape
math separate from allocation policy makes Module 9's later decision explicit.

Questions:

1. Why align from the right, not the left?
2. What is the broadcast of `(2, 1, 4)` and `(3, 1)`?
3. Should `(2, 3)` and `(4, 3)` error, and at which dimension?

**Test:** Several compatible and incompatible pairs, with the expected result or
the expected error.

### Assignment 5.2: Make Binary Ops Broadcast ⭐⭐⭐

**Task:** Extend all four Module 3 binary operations to broadcast. Module 5A
will later change how `sub` and `div` are implemented; this gate first proves
their public shape and value semantics independently of that refactor.

Two designs:

- **A (expand then dispatch):** compute the broadcast shape, `expand` both inputs
  to it (Module 2), then run a stride-aware elementwise kernel that uses
  `physical_offset` per input.
- **B (materialize):** copy each input into a contiguous broadcasted buffer, then
  run the existing same-shape kernel.

**Recommended first choice:** Design A. It reuses your view work and avoids
copies, and it forces your kernel to read inputs through strides, which is the
honest version. The kernel needs each input's shape, strides, and offset passed
in. Design B is simpler but teaches less and copies more.

This is the point where the binary kernel grows from "two contiguous arrays" to
"two strided views and one contiguous output". Update `binary.metal` and the
`Command` (or add a richer command) accordingly.

**Test:** `(4, 3) + (3,)` and `(2, 1, 4) * (3, 1)` against an independent
broadcast oracle, then apply the same public matrix to `sub` and `div`. Also
test `() + (2, 3)`, a transposed or shrunk input, an incompatible pair, rank 5,
and a zero-extent input that fails before allocation or dispatch with a clear
unsupported-case error.

## Part B: Reductions

### Mental Model

A reduction combines many inputs into fewer outputs:

```text
sum over all elements        (4, 5) -> ()
sum over an axis             (4, 5) sum axis=1 -> (4,)
```

Elementwise ops are embarrassingly parallel: each output is independent.
Reductions are not. Multiple threads must combine into one output, which needs
coordination (threadgroup memory, atomics, or a tree). This is why a reduction
needs its own execution path while `mean` does not: `mean` is one `sum` launch
plus one scalar arithmetic operation, with no new coordination problem.

### Assignment 5.3: Implement Full Reduce `sum` ⭐⭐

**Task:** Add `sum(a)` that reduces all elements to a scalar tensor.

A simple single-threadgroup implementation is enough. A tree reduction is also
acceptable as an explicit kernel-learning exercise, provided its numerical
behavior is checked against MLX-C and no performance claim is made without a
measurement.

Questions:

1. How does the result shape `()` interact with your `View` constructor from
   Module 2? Did you decide what a scalar view is back in Assignment 2.1?
2. What is the floating-point order-of-summation caveat, and how does it affect
   your epsilon in the oracle test?

**Test:** `sum` of a known vector against MLX-C on a CPU stream, with a tolerance
that accounts for summation order. The result is a rank-0 tensor with shape `()`,
not a length-one vector. Reject a zero-extent input before dispatch until Module
9 defines and implements empty-tensor storage and reduction behavior.

### Assignment 5.4: Implement Axis Reduce And `mean` ⭐⭐⭐

**Task:** Add `sum(a, axis)`, then implement full and axis `mean` from the
corresponding `sum` result and scalar division. Do not add a `mean` kernel.

Use explicit APIs such as `mean(a)`, `sum(a, axis, keepdim = false)`, and
`mean(a, axis, keepdim = false)`. `axis` is a non-negative integer in
`[0, a.rank())`; reject negative and out-of-range axes rather than translating
them. With `keepdim=false`, remove the reduced axis; with `keepdim=true`, retain
it with extent 1. A rank-0 tensor has no valid axis reduction.

Questions:

1. What is the output shape with and without a `keepdim` option? Pick one policy
   and apply it everywhere.
2. Is `mean` division by the reduced count? How is that count represented as a
   rank-0 tensor, and what changes between full and axis mean? Module 5A will
   later replace this composition without changing the public result.
3. How does an axis reduce map threads to outputs when the reduced axis is not
   the innermost one?

**Recommended first choice:** Implement `sum(axis)` first, build `mean` on top of
it, and choose a single `keepdim` policy now. Inconsistent rank policies are a
classic source of silent shape bugs.

**Test:** Full and axis reductions on `(4, 5)`, including axis 0 and axis 1,
against independent oracles. Cover both `keepdim` values, an invalid axis, a
rank-0 axis error, rank 5, and a zero-extent input that rejects before dispatch.
Test `mean` at the public API level; its implementation strategy is invisible
to the oracle.

## Part C: Matmul

### Mental Model

2-D matmul is the smallest op that is its own kernel and its own layout problem:

```text
C[m, n] = sum_k A[m, k] * B[k, n]
```

It is not elementwise and not a simple reduce. Its performance depends entirely
on memory layout, which is exactly why optimizing it is deferred to Module 13.
Here you only need it correct.

Matmul also explains why Module 5A cannot reduce every operation to a maximal
composition. Luminal's frontend
defines `matmul` as broadcast-`mul` over an expanded `(m, n, k)` intermediate
followed by `sum_reduce` over `k` - mathematically identical, but the
intermediate has `m*n*k` elements where the inputs had `m*k + k*n`. A lazy
compiler can fuse that intermediate away before anything is allocated; an
eager runtime would actually allocate and traffic it. Composition that changes
the asymptotic memory cost is exactly what the budget forbids, so `matmul`
stays a kernel.

### Assignment 5.5: Implement 2-D Matmul And A Contiguous Fast Path ⭐⭐⭐

**Task:** Add `matmul(a, b)` for 2-D inputs. Use a straightforward
one-thread-per-output strided kernel as the general path, and optionally select
a tiled contiguous kernel when both inputs satisfy its layout contract.

Rules:

- Validate inner dimensions match.
- Read inputs through their views, including each input's offset and strides, so
  either transposed input still works. The output is always a new contiguous
  view with shape `(a.shape[0], b.shape[1])`.
- A tiled kernel may use threadgroup memory for zero-offset contiguous inputs.
  Test it separately at non-square and tile-boundary shapes so dispatch geometry
  and edge masking cannot hide behind square matrices.
- Require rank exactly two and reject any zero extent before allocation or
  dispatch until Module 9 establishes empty-tensor support.

Questions:

1. Does your kernel handle a transposed `B` (a non-contiguous view) correctly via
   strides, or does it assume contiguous storage?
2. What is the largest size you can test before the naive kernel is too slow to
   be a comfortable test?

**Test:** Matmuls against MLX-C on a CPU stream, including a non-square
tile-boundary case for the contiguous fast path, a transposed left or right
input for the strided path, a rank error, an inner-dimension mismatch, and a
zero-extent rejection.

## Module 5 Checklist

- [ ] 5.1 Implement `broadcast_shapes`.
- [ ] 5.2 Make binary ops broadcast via expand + strided kernel.
- [ ] 5.3 Implement full-reduce `sum`.
- [ ] 5.4 Implement full/axis `sum` and `mean` with one keepdim policy.
- [ ] 5.5 Implement 2-D matmul with tested contiguous and strided-input paths.

## Exit Criteria

You are ready for mandatory Module 5A when:

1. Broadcasted binary ops match independent oracles on asymmetric shapes.
2. Full and axis reductions match independent oracles within a justified tolerance.
3. Both contiguous and strided matmul paths match MLX-C, including a non-square
   tile-boundary case and a transposed input.
4. Full and axis `mean` match independent oracles at the public API.
5. Rank above four, zero extents, incompatible shapes, invalid axes, and bad
   matmul contracts reject before allocation or dispatch.
6. Every new operation and materially different kernel path has a correctness
   test with its oracle recorded in the current module grading.
7. Optimized paths are accepted as learning work when tested independently;
   performance claims still require measurements.
