# Module 5: Broadcasting, Reductions, And Matmul

## Assignment 5.1: Implement `broadcast_shapes`

### Agent Feedback / Grading

Status: partially passed.

The right-aligned broadcasting rule handles positive asymmetric ranks and incompatible dimensions correctly. Rank-0 tensors also pass through this rule successfully in the public scalar-operand tests.

Remaining gate work:

- Add one direct `view::broadcast_shape(view({}), view({2, 3}))` case so the pure shape function documents its scalar contract without relying on an operator test.
- Fix and test zero-extent shape inference. The current `std::max(a, b)` result turns `(0,)` broadcast with `(1,)` into `(1,)`; NumPy-style shape inference must preserve the zero extent as `(0,)`, even though tensor operations reject empty tensors in this module.

## Assignment 5.2: Make Binary Ops Broadcast

### Agent Feedback / Grading

Status: partially passed.

What is good:

- All four public operations run through the ordinary tensor command path; the special scalar command and Metal kernels have been removed.
- Rank-0 tensors are represented as ordinary tensors with shape `()` and broadcast correctly against ranks 0 through 2 in `ops_scalar.cpp`.
- The asymmetric positive matrix in `ops_broadcast.cpp` covers all four operations in both operand orders against MLX-C.
- Dtype, rank, extent, and compatibility checks happen before output allocation inside the operation path.

Remaining gate work:

- Add a public binary-op test with a transposed or nonzero-offset shrunk input. The kernel is stride-aware, but the Module 5 gate requires execution evidence for this path.
- Add public zero-extent rejection evidence. The existing rank-5 `add`/`sub` tests and direct `check_binary` tests establish the shared rank guard, but direct zero-extent validator tests do not prove rejection before allocation or dispatch.
- MLX-C on its CPU stream is accepted as the independent oracle; a duplicate manual CPU implementation is not required.

## Assignment 5.3: Implement Full Reduce `sum`

### Agent Feedback / Grading

Status: passed for implemented behavior; zero-extent boundary work remains shared module gate work.

Full `sum` returns a rank-0 tensor and passes MLX-C comparisons over rank-0 through rank-3 inputs. The threadgroup tree reduction is accepted as intentional kernel-learning work, and MLX-C on its CPU stream is the project-standard numerical oracle.

Also add a public zero-extent rejection test that demonstrates failure before dispatch. The direct validator test is useful unit coverage but does not establish the whole operation ordering.

## Assignment 5.4: Implement Axis Reduce And `mean`

### Agent Feedback / Grading

Status: passed for implemented behavior; zero-extent boundary work remains shared module gate work.

Full and axis `sum` and `mean` pass MLX-C tests. Axis 0 and axis 1 of a `(4, 5)` tensor are covered with both `keepdim=false` and `keepdim=true`; invalid axes and rank-0 axis reduction are rejected. `mean` is correctly composed from `sum` and division by a rank-0 tensor, with no dedicated mean kernel.

MLX-C on its CPU stream is accepted as the independent oracle, so no duplicate manual reduction implementation is required. Add public zero-extent rejection evidence before allocation or dispatch; rank-5 `mean` already exercises the shared reduction guard.

## Assignment 5.5: Implement Naive 2-D Matmul

### Agent Feedback / Grading

Status: partially passed.

What is good:

- Ordinary and non-contiguous matmul cases pass MLX-C comparisons, including transposed left, right, and both inputs.
- The strided kernel reads through view offsets and strides with one thread per output and a direct loop over `k`.
- Dtype, positive extents, rank two, and inner-dimension compatibility are validated before output allocation in `tensor::matmul`.
- The rectangular dispatch grid now maps columns to `x` and rows to `y`, and the contiguous fast path is restricted to zero-offset views.

Remaining gate work:

- The tiled `16 x 16` contiguous path is accepted as intentional kernel-optimization learning. Add a non-square tile-boundary case such as `(17, 19) @ (19, 33)` so different row/column tile counts and partial tiles are tested. The existing transposed cases exercise the general strided path.
- MLX-C on its CPU stream is accepted as the independent oracle; a duplicate CPU triple loop is not required.
- Add public bad-rank and zero-extent rejection evidence. Direct validator tests do not prove public API ordering.

## Module 5 Verdict

Status: not passed yet.

This worktree compiles and all current tests pass. The scalar refactor is a clear architectural improvement: a scalar is now just a rank-0 tensor, and binary dispatch has one command and one kernel family. The course now accepts MLX-C as the sole independent numerical oracle and accepts the tiled contiguous matmul path as explicit optimization-learning work. The gate remains blocked only by targeted behavioral and coverage gaps.

The minimum remaining work is:

1. Add one non-contiguous public broadcast test.
2. Fix zero-extent shape inference and move zero-extent rejection ahead of storage allocation, then test that public boundary.
3. Add a public matmul rank-error test and a non-square tile-boundary fast-path test.

Verification:

```text
compiled        -> passed
full CTest      -> passed, 40/40
smoke ran       -> passed through Metal-backed CTest cases
gate passed     -> no
```

Commands run:

```text
cmake --build build -j 6
ctest --test-dir build --output-on-failure
git diff --check
```

Do not start Module 5A yet. The remaining items are small, targeted correctness and path-coverage tasks, not broad safety hardening.
