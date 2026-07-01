# comtam Course

This course is a self-guided path for building `comtam`, a tiny eager-mode deep
learning framework in C++20 for Apple devices, with Metal as the only runtime
dependency.

The point is not to copy PyTorch or Magnetron. The point is to rebuild enough of
their ideas that you understand why those ideas exist, and to feel the cost of
each abstraction before you adopt it.

## Course Thesis

comtam stays runtime-first and eager during this course.

That means the main path is:

```text
Metal context
  -> storage and tensor metadata
  -> views and strides
  -> eager operator dispatch
  -> broadcasting, reductions, matmul
  -> reverse-mode autograd
  -> nn modules and SGD
  -> small training examples
  -> hardening only after correctness
```

A tensor op runs now. There is no lazy graph, no scheduler, and no fusion. Each
op validates, allocates a result, optionally records autograd metadata, submits
one Metal command, and returns. This is the opposite of a compiler-first design,
and that is on purpose: you learn runtime ownership, storage, dispatch, and
backend boundaries before you learn graph optimization.

Lazy runtimes and compilers (banhxeo, tinygrad) are still useful, but as
contrast. They teach scheduling and codegen. They should not pull this course
toward graph internals until the eager path is correct and trainable.

## Non-Goals During The Course

These stay out until a failing test or a measurement forces them in. The
reasoning lives in [`../AVOID.md`](../AVOID.md).

- Python bindings.
- Multiple backends or a backend registry.
- Many dtypes (float32 first).
- Lazy graphs, scheduling, fusion.
- Allocator caches.
- Serialization, image/audio IO.
- Profiling counters before correctness.
- Logging or macro infrastructure before it is earned.

## How To Use This Course

1. Read the assignment.
2. Try it yourself for 30-60 minutes.
3. Write down what confused you.
4. Check hints only after you have a concrete question.
5. Implement the smallest working version.
6. Add a test or a trace note.
7. Compare with Magnetron only after your own attempt.

Difficulty ratings:

- ⭐ Straightforward
- ⭐⭐ Requires careful thinking
- ⭐⭐⭐ Challenging, multiple approaches possible

## Recommended Order

Follow this order even if a later module looks shinier.

1. [`MODULE_1.md`](MODULE_1.md) - storage invariants and tensor metadata ownership
2. [`MODULE_2.md`](MODULE_2.md) - master views, strides, offsets, and indexing
3. [`MODULE_3.md`](MODULE_3.md) - eager operator dispatch to a Metal kernel
4. [`MODULE_4.md`](MODULE_4.md) - build forward correctness against a CPU oracle
5. [`MODULE_5.md`](MODULE_5.md) - broadcasting, reductions, and matmul
6. [`MODULE_6.md`](MODULE_6.md) - reverse-mode autograd
7. [`MODULE_7.md`](MODULE_7.md) - nn modules, parameters, and SGD
8. [`MODULE_8.md`](MODULE_8.md) - small end-to-end training examples
9. [`MODULE_9.md`](MODULE_9.md) - hardening after correctness
10. [`PROJECTS.md`](PROJECTS.md) - choose an end-to-end project

Why this order:

- Module 1 is the first real gate because storage ownership and byte sizing
  poison every later lifetime question if they stay vague.
- Module 2 comes before dispatch because wrong views produce wrong kernels that
  look like correct code.
- Module 3 makes one op run end to end before you generalize it.
- Module 4 checks forward semantics before backward rules enter the picture.
- Module 5 adds the shape-changing ops that autograd will depend on.
- Module 6 builds autograd after forward behavior has a reference.
- Modules 7 and 8 prove the whole stack can actually train.
- Module 9 is last because no optimization is allowed before a failing
  performance measurement.

Reference material:

- [`APPENDIX.md`](APPENDIX.md) - Magnetron code map, metal-cpp tips, checklists

## Module Map

| Module | Topic | Main Question |
| --- | --- | --- |
| [`MODULE_1.md`](MODULE_1.md) | Storage invariants | Who owns GPU memory, how are bytes counted, and can storage be shared safely? |
| [`MODULE_2.md`](MODULE_2.md) | Views and strides | How does logical indexing map to storage? |
| [`MODULE_3.md`](MODULE_3.md) | Eager dispatch | What happens from `add(a, b)` to a launched kernel? |
| [`MODULE_4.md`](MODULE_4.md) | Forward correctness | Does forward match a CPU reference? |
| [`MODULE_5.md`](MODULE_5.md) | Broadcast/reduce/matmul | How do shape-changing ops dispatch kernels? |
| [`MODULE_6.md`](MODULE_6.md) | Autograd | Do backward rules compose correctly? |
| [`MODULE_7.md`](MODULE_7.md) | nn and SGD | How do parameters, modules, and optimizers fit eager autograd? |
| [`MODULE_8.md`](MODULE_8.md) | Training | Can comtam learn a real task end to end? |
| [`MODULE_9.md`](MODULE_9.md) | Hardening | Which optimizations are justified by measurement? |

## Progress

| Module | Status | Evidence | Notes |
| --- | --- | --- | --- |
| [`MODULE_1.md`](MODULE_1.md) | Passed | `cmake -S . -B build -DCOMTAM_BUILD_TESTS=ON`; `cmake --build build`; `ctest --test-dir build --output-on-failure` | Verified on 2026-06-30. Detailed grading lives in [`../solution/MODULE_1.md`](../solution/MODULE_1.md). |
| [`MODULE_2.md`](MODULE_2.md) | Passed | `cmake --build build`; `ctest --test-dir build --output-on-failure` | Verified on 2026-06-30. Assignment 2.9 is deferred/non-blocking; detailed grading lives in [`../solution/MODULE_2.md`](../solution/MODULE_2.md). |
| [`MODULE_3.md`](MODULE_3.md) | Passed | `cmake --build build`; `ctest --test-dir build --output-on-failure` | Verified on 2026-07-01. Detailed grading lives in [`../solution/MODULE_3.md`](../solution/MODULE_3.md). |

Keep this table as the compact course progress tracker. Assignment-level feedback, pass/fail reasoning, and missing-test notes belong in the matching file under `docs/solution/`.
