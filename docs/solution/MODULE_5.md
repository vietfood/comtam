# Module 5: Broadcasting, Reductions, And Matmul

## Assignment 5.1: Implement `broadcast_shapes`

### Agent Feedback / Grading

Status: mostly passed for Assignment 5.1.

What is good:

- You started in the right layer: broadcasting shape inference should be pure
  view/shape logic before it is wired into Tensor ops.
- Positive tests now exist for `(3, 1)` with `(2, 1, 4) -> (2, 3, 4)`.
- A positive test now covers the true rank-1 case `(3,)` with `(4, 3)`.
- Incompatible-shape tests exist for `(2, 3)` with `(4, 3)`, and for a
  mismatched middle dimension.

Blocking issues:

- There is still no scalar/rank-0 broadcasting test.
- The implementation uses `size_t a` and `size_t b` for dimensions that come
  from `ViewInt`. Today all shapes should be non-negative, but the shape rule
  should stay in the same signed domain as `View`.

Suggested next step:

Make this a function over `ViewVector` first, returning `ViewVector`. Then let
Tensor/view code wrap that result if needed. That keeps the rule small and
prevents strides/contiguity from leaking into shape inference.

## Assignment 5.2: Make Binary Ops Broadcast

### Agent Feedback / Grading

Status: passed for the required broadcasted binary-op behavior.

What is good:

- `Tensor::binop` now computes a broadcast shape and expands both inputs before
  dispatch.
- A `ViewInfo` payload is the right direction for moving shape/stride/offset
  metadata into Metal.
- The Metal source now compiles into `default.metallib`.
- Runtime loading now uses `default.metallib`, so `#include "utils.h"` no
  longer breaks `Context` construction.
- The command now carries separate input `ViewInfo`s and `Tensor::binop`
  correctly allocates the output with `final_shape`. That fixes two earlier
  design issues.
- Same-shape binary ops pass the existing MLX oracle tests through the new
  view-info kernel path.
- Broadcasted binary ops now run against MLX on the required patterns,
  including `(4, 3)` with `(3,)` and `(2, 1, 4)` with `(3, 1)`, in both operand
  orders.

Caveat:

- Binary ops still explicitly reject non-contiguous inputs before expanding.
  That is acceptable as a temporary policy if documented, but later Module 5
  matmul expects strided input support.

## Assignment 5.3: Implement Full Reduce `sum`

### Agent Feedback / Grading

Status: not started.

There is no public Tensor reduction API, reduction command, reduction Metal
kernel, or correctness test for `sum(a) -> scalar`.

## Assignment 5.4: Implement Axis Reduce And `mean`

### Agent Feedback / Grading

Status: not started.

There is no axis reduction API, no chosen `keepdim` policy, and no `mean`
implementation or tests.

## Assignment 5.5: Implement Naive 2-D Matmul

### Agent Feedback / Grading

Status: mostly passed for contiguous 2-D matmul; one required coverage case is missing.

What is good:

- `Tensor::matmul` exists and validates 2-D inputs plus inner-dimension match.
- `matmul_fp32` is wired through a dedicated `submit_matmul` path.
- A forward test compares matmul against MLX.
- The naive kernel now computes the correct dot product and passes MLX oracle
  tests for contiguous inputs.
- The launch uses CUDA-style threadgroup indexing through
  `dispatchThreadgroups`, while still staying naive: one thread computes one
  output element, with no tiling or threadgroup memory.

Blocking issues:

- The required transposed-input matmul case is not covered yet.

## Module 5 Verdict

Status: not passed.

Verification:

```text
compiled        -> passed with `cmake --build build`
smoke ran       -> passed with `ctest --test-dir build --output-on-failure`
gate passed     -> failed
```

Commands run:

```text
cmake --build build
ctest --test-dir build --output-on-failure
```

Build passed. CTest passed with 27/27 tests. Broadcasted binary ops and
contiguous matmul now match MLX oracles.

Do not start Module 6 yet. The honest next checkpoint is: add the required
transposed-input matmul case, then implement reductions.
