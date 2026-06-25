## Module 5: Broadcasting, Reductions, And Matmul

Add the shape-changing ops that real models need, and that autograd will depend
on.

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
oracle.

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
approach. Correctness first; the tree reduction is a Module 9 concern, and only
after a measurement.

Questions:

1. How does the result shape `()` interact with your `View` constructor from
   Module 2? Did you decide what a scalar view is back in Assignment 2.1?
2. What is the floating-point order-of-summation caveat, and how does it affect
   your epsilon in the oracle test?

**Test:** `sum` of a known vector against a CPU sum, with a tolerance that
accounts for summation order.

### Assignment 5.4: Implement Axis Reduce And `mean` ⭐⭐⭐

**Task:** Add `sum(a, axis)` and `mean(a, axis)`.

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
oracles.

## Part C: Matmul

### Mental Model

2-D matmul is the smallest op that is its own kernel and its own layout problem:

```text
C[m, n] = sum_k A[m, k] * B[k, n]
```

It is not elementwise and not a simple reduce. Its performance depends entirely
on memory layout, which is exactly why optimizing it is deferred to Module 9.
Here you only need it correct.

### Assignment 5.5: Implement Naive 2-D Matmul ⭐⭐⭐

**Task:** Add `matmul(a, b)` for 2-D inputs with a straightforward kernel: one
thread per output element, looping over `k`.

Rules:

- Validate inner dimensions match.
- Do not tile, do not use threadgroup memory, do not optimize. The naive kernel
  is the oracle's friend.
- Read inputs through their views so a transposed input still works.

Questions:

1. Does your kernel handle a transposed `B` (a non-contiguous view) correctly via
   strides, or does it assume contiguous storage?
2. What is the largest size you can test before the naive kernel is too slow to
   be a comfortable test?

**Test:** Small matmuls (for example `(2, 3) @ (3, 4)`) against a CPU triple
loop, including one case with a transposed input.

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
