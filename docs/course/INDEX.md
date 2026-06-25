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

1. [`MODULE_0.md`](MODULE_0.md) - understand the current codebase and Metal context
2. [`MODULE_1.md`](MODULE_1.md) - storage and tensor metadata ownership
3. [`MODULE_2.md`](MODULE_2.md) - master views, strides, offsets, and indexing
4. [`MODULE_3.md`](MODULE_3.md) - eager operator dispatch to a Metal kernel
5. [`MODULE_4.md`](MODULE_4.md) - build forward correctness against a CPU oracle
6. [`MODULE_5.md`](MODULE_5.md) - broadcasting, reductions, and matmul
7. [`MODULE_6.md`](MODULE_6.md) - reverse-mode autograd
8. [`MODULE_7.md`](MODULE_7.md) - nn modules, parameters, and SGD
9. [`MODULE_8.md`](MODULE_8.md) - small end-to-end training examples
10. [`MODULE_9.md`](MODULE_9.md) - hardening after correctness
11. [`PROJECTS.md`](PROJECTS.md) - choose an end-to-end project

Why this order:

- Module 0 forces an honest map of what already exists before you build on it.
- Module 1 settles ownership early because a confused owner poisons every later
  lifetime question.
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
- [`BUG_LEDGER.md`](BUG_LEDGER.md) - concrete bugs discovered while running the course

## Module Map

| Module | Topic | Main Question |
| --- | --- | --- |
| [`MODULE_0.md`](MODULE_0.md) | Current codebase | What happens when a tensor moves host -> GPU -> host? |
| [`MODULE_1.md`](MODULE_1.md) | Storage and metadata | Who owns GPU memory and when is it freed? |
| [`MODULE_2.md`](MODULE_2.md) | Views and strides | How does logical indexing map to storage? |
| [`MODULE_3.md`](MODULE_3.md) | Eager dispatch | What happens from `add(a, b)` to a launched kernel? |
| [`MODULE_4.md`](MODULE_4.md) | Forward correctness | Does forward match a CPU reference? |
| [`MODULE_5.md`](MODULE_5.md) | Broadcast/reduce/matmul | How do shape-changing ops dispatch kernels? |
| [`MODULE_6.md`](MODULE_6.md) | Autograd | Do backward rules compose correctly? |
| [`MODULE_7.md`](MODULE_7.md) | nn and SGD | How do parameters, modules, and optimizers fit eager autograd? |
| [`MODULE_8.md`](MODULE_8.md) | Training | Can comtam learn a real task end to end? |
| [`MODULE_9.md`](MODULE_9.md) | Hardening | Which optimizations are justified by measurement? |

## Module Gates

This course is graded by gates, not by demos. A printed array that looks
plausible is not a pass.

- Do not start module N+1 until module N's gate is met.
- Every new op needs a correctness test against a CPU oracle.
- Every autograd rule needs a gradient test.
- Every Metal object needs one obvious owner.
- No hidden global framework state.
- No optimization before a failing performance measurement.

When you grade yourself, separate three claims explicitly:

```text
compiled        -> the build succeeded
smoke ran       -> the executable produced output without crashing
gate passed     -> the module's exit criteria are met with tests
```

## Ground Rules

- Keep patches small.
- Do not mix refactors with behavior changes.
- Every functionality change needs a test or a documented trace.
- Prefer one clear implementation over a clever general one.
- Every Metal object has exactly one owner. If you cannot name the owner, stop.
- If you cannot explain what a kernel launch does, pause and trace it.
- Treat printed arrays as debugging, not as tests.
