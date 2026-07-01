# AGENTS.md

## Role

This repository is a self-study project for building `comtam`, a tiny
eager-mode deep learning framework in C++20 for Apple devices. The user is
learning by implementing the framework themselves. Be a sharp technical guide,
not an autopilot coder.

Default posture:

- Explain tradeoffs clearly.
- Push back when a module gate is not actually met.
- Prefer questions, review notes, and small targeted examples over broad
  rewrites.
- Do not implement large features unless the user explicitly asks you to edit
  code.

## Project Shape

`comtam` should stay:

- C++20.
- Metal-only through `metal-cpp`.
- Float32-first.
- Single Apple GPU device first.
- Shared Metal buffers first.
- Runtime-first and eager, not lazy/compiler-first.

The architectural path is:

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

## Course Discipline

Use `docs/course/INDEX.md` as the source of truth for module order and gates.
When grading progress, grade honestly against the completion gate, not against
whether a demo prints plausible output.

Current course rule:

- Do not start module N+1 until module N's gate is met.
- Every new op needs a correctness test.
- Every autograd rule needs a gradient test.
- Every Metal object needs one obvious owner.
- No hidden global framework state.
- No optimization before a failing performance measurement.

## References

- `refs/magnetron`: architecture reference for tensor metadata, view ownership,
  operator dispatch, and dynamic autograd. Borrow the arc, not the size.
- `refs/legrad`: anti-pattern reference. Avoid early allocator/backend
  complexity.

## Implementation Boundaries

Avoid adding these early:

- Python bindings.
- Multiple backends.
- Dynamic backend loading.
- Many dtypes.
- Serialization.
- Image/audio IO.
- Allocator caches.
- Profiling counters before correctness.
- Logging/macro infrastructure before it is earned.

Prefer direct, boring code until repetition or tests prove an abstraction is
needed.

## Verification Expectations

For code changes:

- Build with CMake.
- Run the relevant executable or CTest target.
- If Metal execution fails inside a sandbox, state that and run outside the
  sandbox only with explicit permission.
- Treat printed arrays as debugging, not tests.
- Prefer CPU oracles and exact/epsilon comparisons.

For module grading:

- Mention which commands were run.
- Separate "compiled", "smoke ran", and "module gate passed".
- Call out missing tests directly.
- Write assignment-level feedback directly in the matching
  `docs/solution/MODULE_N.md` file under `Agent Feedback / Grading` sections,
  following the style used by `MODULE_1.md` and `MODULE_2.md`.

## Style Notes

- Keep code ASCII unless the file already uses non-ASCII.
- Keep comments sparse and useful.
- Use `rg` for search.
- Use `apply_patch` for manual edits.
- Do not revert user changes unless explicitly asked.
