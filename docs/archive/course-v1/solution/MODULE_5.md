# Module 5: Broadcasting, Reductions, And Matmul

## Assignment 5.1: Implement `broadcast_shapes`

### Agent Feedback / Grading

Status: passed.

The right-aligned broadcasting rule handles positive asymmetric ranks, incompatible dimensions, rank-0 tensors, and zero extents correctly. The zero-extent rule is symmetric by construction and now has regression assertions for `(0,)` with `(1,)` in both operand orders.

## Assignment 5.2: Make Binary Ops Broadcast

### Agent Feedback / Grading

Status: passed.

What is good:

- All four public operations run through the ordinary tensor command path; the special scalar command and Metal kernels have been removed.
- Rank-0 tensors are represented as ordinary tensors with shape `()` and broadcast correctly against ranks 0 through 2 in `ops_scalar.cpp`.
- The asymmetric positive matrix in `ops_broadcast.cpp` covers all four operations in both operand orders against MLX-C.
- Dtype, rank, extent, and compatibility checks happen before output allocation inside the operation path.

Remaining gate work:

- The new transposed-input broadcast test passes against MLX-C and exercises the stride-aware public path.
- Public zero-extent operations reject before dispatch. The test correctly reaches them through a zero-extent view over live non-empty storage while preserving the zero-byte allocation rejection.
- MLX-C on its CPU stream is accepted as the independent oracle; no duplicate manual CPU implementation is required.

## Assignment 5.3: Implement Full Reduce `sum`

### Agent Feedback / Grading

Status: passed.

Full `sum` returns a rank-0 tensor and passes MLX-C comparisons over rank-0 through rank-3 inputs. The threadgroup tree reduction is accepted as intentional kernel-learning work, and MLX-C on its CPU stream is the project-standard numerical oracle.

The public zero-extent reduction test rejects before dispatch using a zero-extent view over live storage.

## Assignment 5.4: Implement Axis Reduce And `mean`

### Agent Feedback / Grading

Status: passed.

Full and axis `sum` and `mean` pass MLX-C tests. Axis 0 and axis 1 of a `(4, 5)` tensor are covered with both `keepdim=false` and `keepdim=true`; invalid axes and rank-0 axis reduction are rejected. `mean` is correctly composed from `sum` and division by a rank-0 tensor, with no dedicated mean kernel.

MLX-C on its CPU stream is accepted as the independent oracle, so no duplicate manual reduction implementation is required. Public zero-extent rejection is covered through live backing storage, and rank-5 `mean` exercises the shared reduction guard.

## Assignment 5.5: Implement 2-D Matmul And A Contiguous Fast Path

### Agent Feedback / Grading

Status: passed.

What is good:

- Ordinary and non-contiguous matmul cases pass MLX-C comparisons, including transposed left, right, and both inputs.
- The strided kernel reads through view offsets and strides with one thread per output and a direct loop over `k`.
- Dtype, positive extents, rank two, and inner-dimension compatibility are validated before output allocation in `tensor::matmul`.
- The rectangular dispatch grid now maps columns to `x` and rows to `y`, and the contiguous fast path is restricted to zero-offset views.

Remaining gate work:

- The tiled `16 x 16` contiguous path is accepted as intentional kernel-optimization learning. Multiple non-square and tile-boundary cases now cover different row/column tile counts, partial tiles, and multiple `k` phases. The expanded transposed matrix exercises the general strided path.
- MLX-C on its CPU stream is accepted as the independent oracle; a duplicate CPU triple loop is not required.
- Public rank-1, rank-3, inner-dimension, and zero-extent rejection paths are tested.

## Module 5 Verdict

Status: passed.

This worktree compiles and all 43 tests pass. Non-contiguous broadcasting, symmetric zero-extent shape inference, zero-extent operation rejection, full and axis reductions, public `mean`, tiled matmul boundaries, strided matmul, and public matmul contract errors have the required coverage. MLX-C on its CPU stream is the accepted independent oracle, and both matmul kernel paths are tested. Module 5 is complete.

Verification:

```text
compiled        -> passed
full CTest      -> passed, 43/43
smoke ran       -> passed through Metal-backed CTest cases
gate passed     -> yes
```

Commands run:

```text
cmake --build build -j 6
ctest --test-dir build -R "View broadcast shape" --output-on-failure
ctest --test-dir build --output-on-failure
git diff --check
```

Proceed to mandatory Module 5A before Module 6. Module 5A should stabilize the primitive/composed operator boundary and numerical contracts that autograd will rely on.
