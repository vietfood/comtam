# Module 5: Broadcasting, Reductions, And Matmul

## Assignment 5.1: Implement `broadcast_shapes`

### Agent Feedback / Grading

Status: partially passed. The core right-aligned shape rule works for the tested
positive and incompatible cases, but the required scalar case and zero-extent
policy are not covered yet.

What is good:

- You started in the right layer: broadcasting shape inference should be pure
  view/shape logic before it is wired into Tensor ops.
- Positive tests now exist for `(3, 1)` with `(2, 1, 4) -> (2, 3, 4)`.
- A positive test now covers the true rank-1 case `(3,)` with `(4, 3)`.
- Incompatible-shape tests exist for `(2, 3)` with `(4, 3)`, and for a
  mismatched middle dimension.

Remaining gate work:

- There is no scalar/rank-0 broadcasting test such as `()` with `(2, 3)`.
- There is no zero-extent shape-inference test documenting that shape math may
  infer a zero extent even though Module 5 tensor operations must reject it.
- The implementation uses `size_t a` and `size_t b` for dimensions that come
  from `ViewInt`. Today all shapes should be non-negative, but the shape rule
  should stay in the same signed domain as `View`.

Suggested next step:

Make this a function over `ViewVector` first, returning `ViewVector`. Then let
Tensor/view code wrap that result if needed. That keeps the rule small and
prevents strides/contiguity from leaking into shape inference.

## Assignment 5.2: Make Binary Ops Broadcast

### Agent Feedback / Grading

Status: partially passed. The main expand-plus-strided-kernel path works for the
tested positive cases, but the Assignment 5.2 test matrix is incomplete.

What is good:

- `Tensor::binop` now computes a broadcast shape and expands both inputs before
  dispatch.
- A `ViewInfo` payload is the right direction for moving shape/stride/offset
  metadata into Metal.
- The binary Metal source has been integrated into `default.metallib`, and prior
  verification recorded successful execution through that path.
- The command now carries separate input `ViewInfo`s and `Tensor::binop`
  correctly allocates the output with `final_shape`. That fixes two earlier
  design issues.
- Same-shape binary ops pass the existing MLX oracle tests through the new
  view-info kernel path.
- Broadcasted binary ops now run against MLX on the required patterns,
  including `(4, 3)` with `(3,)` and `(2, 1, 4)` with `(3, 1)`, in both operand
  orders.
- Tensor-level incompatible-shape rejection is covered for all four binary ops.

Remaining gate work:

- Add the required rank-0 tensor operand case, `() + (2, 3)`, against an oracle.
- Add tensor-operation coverage for zero-extent rejection before allocation or
  dispatch.
- The current positive tests use MLX. That is a valid independent framework
  oracle, but this module's written contract specifically requires a CPU oracle.
  Add a small manual CPU broadcast oracle, or revise the contract consistently
  rather than silently counting MLX as different evidence.
- Binary ops reject non-contiguous inputs before expanding. That is narrower
  than the module contract's "strided input views" wording, so either support
  ordinary strided views or explicitly narrow the course contract. The better
  course-aligned choice is to remove the rejection because the kernel already
  receives strides and offsets.

## Assignment 5.3: Implement Full Reduce `sum`

### Agent Feedback / Grading

Status: scaffold started; no assignment behavior is complete yet.

`submit_reduce` and a draft `reduce_sum` kernel exist, but neither implements a
reduction. There is still no public Tensor reduction API or correctness test for
`sum(a) -> scalar`. The current draft also needs to be moved under
`comtam/kernels/` and written as valid Metal before it can join the kernel build.

The in-progress command code currently refers to `Op::REDUCE_SUM` and
`Op::REDUCE_MAX`, while the enum defines `Op::SUM` and `Op::MAX`. Resolve that
naming mismatch before treating the scaffold as compilable.

## Assignment 5.4: Implement Axis Reduce And `mean`

### Agent Feedback / Grading

Status: not started at the public API and test level.

There is no axis reduction API, no chosen `keepdim` policy, and no `mean`
implementation or tests.

## Assignment 5.5: Implement Naive 2-D Matmul

### Agent Feedback / Grading

Status: partially passed. Contiguous 2-D matmul matches the oracle, but the
required layout and validation matrix is incomplete.

What is good:

- `Tensor::matmul` exists and validates 2-D inputs plus inner-dimension match.
- `matmul_fp32` is wired through a dedicated `submit_matmul` path.
- A forward test compares matmul against MLX.
- The naive kernel now computes the correct dot product and passes MLX oracle
  tests for contiguous inputs.
- The launch uses CUDA-style threadgroup indexing through
  `dispatchThreadgroups`, while still staying naive: one thread computes one
  output element, with no tiling or threadgroup memory.

Remaining gate work:

- Add a transposed left or right input test. The kernel uses
  `physical_offset`, so this test should validate the intended strided-input
  design rather than require materialization.
- Add tests for bad rank and mismatched inner dimensions.
- Add explicit zero-extent rejection before output allocation or dispatch.
- The positive tests use MLX, while the module contract asks for a CPU triple
  loop. Add that small manual oracle or change the course-wide oracle policy.

## Module 5 Verdict

Status: not passed.

Verification:

```text
compiled        -> not verified for the current worktree
smoke ran       -> not run because the current build did not complete
gate passed     -> failed
```

Commands run:

```text
cmake --build build
```

The current build stopped while generating `default.metallib` because `xcrun`
could not find the `metallib` utility in the selected Xcode toolchain. This is
an environment/toolchain failure, so it does not prove a C++ or Metal source
failure, but the current reduction scaffold also contains the enum-name mismatch
noted in Assignment 5.3 and has not been verified.

Prior grading recorded 27/27 tests passing for broadcasted binary ops and
contiguous matmul. That historical result is useful regression evidence, but it
is not current-worktree verification and it does not satisfy the Module 5 gate.

Two course-contract issues also need an explicit decision before this module is
published as course material. First, command and Metal view metadata silently
cap rank at four, while the Module 5 supported scope states no rank limit;
either validate and document rank `<= 4` or remove the fixed-rank truncation.
Second, the Module 4 correctness map still labels broadcasting and matmul as
unsupported, so Exit Criterion 4 is not met even for the implemented cases.

Do not start Module 6 yet. Finish the missing broadcast edge cases and matmul
validation/layout cases, implement full and axis reductions plus `mean`, update
the Module 4 correctness map, then rerun the build and complete CTest suite.
