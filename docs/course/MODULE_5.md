## Module 5: Broadcasting, Reductions, And Matmul

Add the shape-changing ops that real models need, and that autograd will depend
on.

## Module Contract

**Prerequisite:** Module 4 has independent forward-oracle coverage and a
current correctness map. **Deliverables:** `broadcast_shapes`, broadcasted
float32 binary ops using strided input views, full and axis `sum`, axis `mean`,
and naive 2-D `matmul` that reads strided inputs. Every new operation requires a
CPU-oracle correctness test and an updated Module 4 correctness map.

Supported scope is right-aligned NumPy-style broadcasting, rank-0 scalar
tensors, positive-sized dimensions, non-negative axes, and 2-D matmul only.
Zero-sized dimensions must reject clearly in this module because the current
storage layer forbids zero-byte allocation; Module 9 revisits empty-tensor
semantics deliberately. Negative axes, batched matmul, dtype promotion, implicit
materialization, tiling, threadgroup-memory optimization, and fusion are also
unsupported. Completion evidence is CTest coverage of the matrix below and the
simplest correct kernels; optimization remains out of scope even if a benchmark
is slow.

| Area | Required evidence |
| --- | --- |
| Broadcast binary ops | compatible asymmetric ranks, scalar operand, incompatible shapes, and zero-extent rejection |
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
layout. Get these right with oracles now, because Module 6 builds gradients on
top of them.

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

**Task:** Extend the Module 3 binary ops to broadcast.

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

**Test:** `(4, 3) + (3,)` and `(2, 1, 4) * (3, 1)` against a CPU broadcast
oracle. Also test `() + (2, 3)`, an incompatible pair, and a zero-extent input
that fails before allocation or dispatch with a clear unsupported-case error.

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
is a different kind of kernel.

### Assignment 5.3: Implement Full Reduce `sum` ⭐⭐

**Task:** Add `sum(a)` that reduces all elements to a scalar tensor.

Start with the simplest correct kernel, even a single-threadgroup or naive
approach. Correctness first; the tree reduction is a Module 13 concern, and only
after a measurement.

Questions:

1. How does the result shape `()` interact with your `View` constructor from
   Module 2? Did you decide what a scalar view is back in Assignment 2.1?
2. What is the floating-point order-of-summation caveat, and how does it affect
   your epsilon in the oracle test?

**Test:** `sum` of a known vector against a CPU sum, with a tolerance that
accounts for summation order. The result is a rank-0 tensor with shape `()`,
not a length-one vector. Reject a zero-extent input before dispatch until Module
9 defines and implements empty-tensor storage and reduction behavior.

### Assignment 5.4: Implement Axis Reduce And `mean` ⭐⭐⭐

**Task:** Add `sum(a, axis)` and `mean(a, axis)`.

Use an explicit API such as `sum(a, axis, keepdim = false)` and
`mean(a, axis, keepdim = false)`. `axis` is a non-negative integer in
`[0, a.rank())`; reject negative and out-of-range axes rather than translating
them. With `keepdim=false`, remove the reduced axis; with `keepdim=true`, retain
it with extent 1. A rank-0 tensor has no valid axis reduction.

Questions:

1. What is the output shape with and without a `keepdim` option? Pick one policy
   and apply it everywhere.
2. Is `mean` just `sum` divided by the reduced count? Where should that division
   happen, host or kernel?
3. How does an axis reduce map threads to outputs when the reduced axis is not
   the innermost one?

**Recommended first choice:** Implement `sum(axis)` first, build `mean` on top of
it, and choose a single `keepdim` policy now. Inconsistent rank policies are a
classic source of silent shape bugs.

**Test:** Axis reductions on `(4, 5)` over axis 0 and axis 1, against CPU
oracles. Cover both `keepdim` values, an invalid axis, a rank-0 axis error, and a
zero-extent input that rejects before dispatch.

## Part C: Matmul

### Mental Model

2-D matmul is the smallest op that is its own kernel and its own layout problem:

```text
C[m, n] = sum_k A[m, k] * B[k, n]
```

It is not elementwise and not a simple reduce. Its performance depends entirely
on memory layout, which is exactly why optimizing it is deferred to Module 13.
Here you only need it correct.

### Assignment 5.5: Implement Naive 2-D Matmul ⭐⭐⭐

**Task:** Add `matmul(a, b)` for 2-D inputs with a straightforward kernel: one
thread per output element, looping over `k`.

Rules:

- Validate inner dimensions match.
- Do not tile, do not use threadgroup memory, do not optimize. The naive kernel
  is the oracle's friend.
- Read inputs through their views, including each input's offset and strides, so
  either transposed input still works. The output is always a new contiguous
  view with shape `(a.shape[0], b.shape[1])`.
- Require rank exactly two and reject any zero extent before allocation or
  dispatch until Module 9 establishes empty-tensor support.

Questions:

1. Does your kernel handle a transposed `B` (a non-contiguous view) correctly via
   strides, or does it assume contiguous storage?
2. What is the largest size you can test before the naive kernel is too slow to
   be a comfortable test?

**Test:** Small matmuls (for example `(2, 3) @ (3, 4)`) against a CPU triple
loop, including a transposed left or right input, a rank error, an
inner-dimension mismatch, and a zero-extent rejection.

## Module 5 Checklist

- [ ] 5.1 Implement `broadcast_shapes`.
- [ ] 5.2 Make binary ops broadcast via expand + strided kernel.
- [ ] 5.3 Implement full-reduce `sum`.
- [ ] 5.4 Implement axis `sum`/`mean` with one keepdim policy.
- [ ] 5.5 Implement naive 2-D matmul, strided-input aware.

## Exit Criteria

You are ready for Module 6 when:

1. Broadcasted binary ops match a CPU oracle on asymmetric shapes.
2. Full and axis reductions match CPU oracles within a justified tolerance.
3. Naive matmul matches a CPU triple loop, including a transposed input.
4. Every new op has a correctness test, and the correctness map from Module 4 is
   updated.
5. No optimization has been added; every kernel is the simplest correct version.
