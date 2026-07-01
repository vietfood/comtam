# Module 4: Forward Correctness

The implementation evidence currently lives in:

- `tests/support/forward_compare.h`
- `tests/support/mlx_oracle.h`
- `tests/tensor/forward.cpp`
- `tests/tensor/operations.cpp`

That is the right place for the code. The table below records what is verified,
what is intentionally unsupported until Module 5, and which tests provide the
evidence.

## Assignment 4.1: Build A Forward Comparison Helper

### Agent Feedback / Grading

Status: passed.

What is good:

- `tests/support/forward_compare.h` provides one shared comparison path for
  forward tests instead of letting every test invent its own tolerance and shape
  checks.
- `require_shape_matches` checks rank, `numel`, and logical shape before value
  comparison.
- `require_forward_matches` uses view-aware `Tensor::to_vector`, so movement
  tests exercise the actual readback semantics from Module 2.
- The helper supports exact and approximate value modes, which matches the
  course rule: exact for movement/copy cases, epsilon comparisons for float
  arithmetic.
- MLX stays test-only through `tests/support/mlx_oracle.h` and the `comtam_tests`
  target. It does not become a `comtam_lib` dependency.

Evidence:

- `Forward compare helper` covers an exact host round-trip and an approximate
  arithmetic comparison.
- CTest discovers and runs the helper test.

## Assignment 4.2: Test Elementwise Ops

### Agent Feedback / Grading

Status: passed.

What is good:

- `Forward compare vs MLX for elementwise ops` checks `add/sub/mul/div` against
  MLX C through the reusable comparison helper.
- `tests/tensor/operations.cpp` also checks the binary ops against MLX C on
  several asymmetric shapes.
- Division uses random positive RHS values, so it tests normal division rather
  than accidentally turning into divide-by-zero policy.

Evidence:

- `Forward compare vs MLX for elementwise ops` covers the required Module 4
  shapes `(2, 3)`, `(3, 1)`, `(1, 4)`, and `(2, 3, 4)`.
- It also covers `(3, 4)`, which is useful extra asymmetric coverage.

## Assignment 4.3: Test Movement Ops Through Readback

### Agent Feedback / Grading

Status: passed.

What is good:

- `Forward compare vs MLX for movement ops` covers transpose, general permute,
  shrink/slice, expand, reshape, and a chained slice-then-transpose view.
- The movement tests compare view-aware `to_vector` results against MLX C, so
  they verify values through the actual logical-to-physical readback path.
- There is at least one direct non-contiguous readback case.

Evidence:

- Each movement section in `tests/tensor/forward.cpp` now records the expected
  comtam view state before checking values against MLX C.
- The comments cover shape, strides, and offset for transpose, permute,
  shrink/slice, expand, reshape, and chained slice-then-transpose.

## Assignment 4.4: Write Down The Correctness Map

### Agent Feedback / Grading

Status: passed.

The correctness map is the main reason a one-line solution is not enough. The
tests are evidence, and this table is the compact contract that future modules
can rely on.

Correctness map:

```text
op / behavior              status          oracle       note
host round-trip            verified        manual       exact values
add/sub/mul/div            verified        mlx-c        (2,3), (3,1), (1,4), (3,4), (2,3,4)
transpose                  verified        mlx-c        view-aware readback
permute                    verified        mlx-c        view-aware readback
shrink / slice             verified        mlx-c        view-aware readback
expand view                verified        mlx-c        stride-0 readback
reshape contiguous         verified        mlx-c        contiguous-only path
slice then transpose       verified        mlx-c        chained view
broadcasted binary op      unsupported     -           Module 5, rejected today by exact-shape checks
reductions                 unsupported     -           Module 5
matmul                     unsupported     -           Module 5
```

Notes:

- Broadcasted binary ops, reductions, and matmul do not have public supported
  APIs yet. They are intentionally deferred to Module 5.
- Current binary ops reject mismatched shapes, so unsupported broadcasting is
  explicit in behavior rather than a silent gap.

## Module 4 Verdict

Status: passed.

Verification:

```text
compiled        -> passed with `cmake --build build`
smoke ran       -> passed with `ctest --test-dir build --output-on-failure`
gate passed     -> passed
```

Commands run:

```text
cmake --build build
ctest --test-dir build --output-on-failure
```

CTest passed with 24/24 tests.

What is solid:

- The reusable forward comparison helper exists and runs under CTest.
- MLX C is integrated as a test-only independent oracle.
- Elementwise and movement tests are running against MLX C.
- The view-aware readback path is exercised by movement tests.
- The correctness map records verified behavior and Module 5 gaps.
