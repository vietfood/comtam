## Module 4: Forward Correctness

Make comtam prove its forward behavior against a trusted reference before
autograd enters the picture.

## Why This Module Comes Before Autograd

Autograd depends on forward semantics.

If `expand`, `slice`, `permute`, or elementwise math is wrong, gradient tests
will fail in noisy ways and you will debug backward formulas when the forward
value was already broken.

comtam has no Python and no PyTorch in-process. The reference here is a
test-only oracle: either a tiny manual/CPU computation for simple cases, or MLX
C for framework-level semantics such as transpose, slice, broadcast, reshape,
and elementwise math. MLX is not a comtam backend and must not become a
`comtam_lib` dependency. It lives in tests as an independent reference.

This module builds that harness. Printed arrays are debugging, not tests.

## What "Oracle" Means Here

An oracle is the smallest independent computation that tells you the GPU answer
is right.

```text
manual      a hand-written expected vector for a tiny fixed input
cpu kernel  a plain C++ loop computing the same op on host memory
mlx-c       an external framework reference, called from tests on a CPU stream
```

MLX C is the workhorse once shape semantics get interesting. It is independent
of comtam's Metal path, so a disagreement points at comtam's GPU dispatch,
view/readback math, or at a mismatch in how the test translated the operation to
MLX. Keep the wrapper tiny and boring: create arrays, run the MLX op, force a
contiguous result, synchronize, read back float32, and free every MLX handle.

Manual and plain C++ CPU oracles are still allowed when they are clearer than an
external dependency. Do not use printed arrays as evidence.

## Assignment 4.1: Build A Forward Comparison Helper ⭐⭐

**Task:** Add a reusable test helper that compares a comtam result against an
independent oracle. Prefer the MLX C helper for framework semantics, and keep
plain C++ or hand-written expected vectors for very small cases where that is
clearer.

It should:

- create identical host inputs for comtam and the oracle
- run the comtam op and read it back with the view-aware `to_vector` from
  Module 2
- run the oracle through a small helper, not raw MLX handles scattered through
  test cases
- compare shape first, then values
- use exact comparison for copies and movement ops, and a small epsilon
  (for example `1e-5`) for float arithmetic

The MLX C notes live in [`../MLX_C_ORACLE.md`](../MLX_C_ORACLE.md). The helper
should stay under `tests/`, wired into CTest, and linked only into test targets.
Keep it small.

**Why this assignment:** A shared comparison standard stops every test from
inventing its own notion of "close enough".

## Assignment 4.2: Test Elementwise Ops ⭐⭐

**Task:** Cover the Module 3 binary ops thoroughly.

Test groups:

- elementwise: add, sub, mul, div
- input shapes: `(2, 3)`, `(3, 1)`, `(1, 4)`, `(2, 3, 4)`

Avoid only testing `(2, 2)`. Symmetric shapes hide axis-order and stride bugs.

For division, include a case that exercises non-trivial values (no divide by
zero yet, that is a hardening concern).

**Why this assignment:** Elementwise ops are the baseline. If these are not
trusted, every later op built on the same dispatch path inherits the doubt.

## Assignment 4.3: Test Movement Ops Through Readback ⭐⭐⭐

**Task:** Verify the Module 2 view ops by reading them back and comparing to an
independent implementation of the same movement. MLX C is a good fit here
because it already defines transpose, slice, broadcast, and reshape semantics.

Test:

- transpose and general permute
- shrink / slice
- expand (broadcast)
- reshape (contiguous case)
- one chained view, for example `slice` then `transpose`

For each non-obvious case, record the expected comtam `shape`, `strides`, and
`offset` in the test as a comment, then assert values against the oracle.

Include at least one test where a non-contiguous view is read back directly.
This is the test that catches stride bugs immediately.

**Why this assignment:** Movement ops are where comtam stops being "arrays with
operators" and becomes a storage-interpretation system. The readback test is the
proof.

## Assignment 4.4: Write Down The Correctness Map ⭐

**Task:** Produce a short table of what is verified, what is unsupported, and
what is a known gap.

```text
op            status        oracle        note
add           verified      mlx-c         asymmetric shapes
transpose     verified      mlx-c         via view-aware to_vector
broadcast op  unsupported   -             Module 5
matmul        unsupported   -             Module 5
```

Anything "unsupported" should become an explicit failing or skipped test so it
cannot be silently forgotten.

**Why this assignment:** A written correctness map is the difference between "I
think it works" and a gate you can defend. Module 6 will start from this map.

## Module 4 Checklist

- [ ] 4.1 Build a forward comparison helper wired into CTest.
- [ ] 4.2 Test elementwise ops on asymmetric shapes.
- [ ] 4.3 Test movement ops through view-aware readback.
- [ ] 4.4 Write the correctness map and mark unsupported ops explicitly.

## Exit Criteria

You are ready for Module 5 when:

1. A reusable oracle comparison helper exists and runs under CTest.
2. Elementwise and movement ops pass on asymmetric and non-contiguous shapes.
3. Unsupported ops are explicit skipped/failing tests, not silent gaps.
4. You have a written correctness map you trust enough to build autograd on.
