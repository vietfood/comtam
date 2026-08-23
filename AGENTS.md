# AGENTS.md

## Role

This repository is a self-study project for building `comtam`, a tiny eager-mode deep learning framework in C++20 for Apple devices. The user is learning by implementing the framework themselves. Be a sharp technical guide, not an autopilot coder.

Default posture:

- Explain tradeoffs clearly.
- Push back when a chapter gate is not actually met.
- Prefer questions, review notes, and small targeted examples over broad rewrites.
- Do not implement large features unless the user explicitly asks you to edit code.

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

Use `docs/course/INDEX.md` as the baseline for track and chapter order. The active course is organized around difficult mechanisms, not numbered modules; a mechanism may span several chapters. Reconcile the course with verified source code, tests, and explicit user decisions before grading; do not treat stale chapter text as stronger evidence than the current worktree. The source does not silently redefine its own acceptance criteria, so record deliberate scope or learning-goal changes in the course before applying the revised gate.

Adapt course difficulty to the user's demonstrated competence and learning goals. Preserve correctness and architectural prerequisites, but allow a harder implementation when it teaches a relevant topic and its additional execution paths are tested independently.

Treat the course as living guidance rather than a mechanism for forcing the source back to an older design. When the current source contains a deliberate, technically justified, tested design that is better suited to the user's learning goal, update the course and grading contract to describe that design instead of blocking progress solely for textual conformance. Existing code is not self-justifying: require a concrete mechanism, tradeoff, and relevant test evidence before calling a divergence better.

Current course rule:

- Do not start chapter N+1 until chapter N's completion gate is met.
- Every new op needs a correctness test.
- Every autograd rule needs a gradient test.
- Every Metal object needs one obvious owner.
- No hidden global framework state.
- Optimization exercises are allowed as explicit learning work when the simple semantics remain independently verified. Do not make performance claims or add production optimization infrastructure before measurement.

## References

External reference implementations are no longer vendored. Distilled notes live in `docs/refs/`, each citing upstream paths:

- `docs/refs/MAGENETRON.md`: architecture reference for tensor metadata, view ownership, operator dispatch, and dynamic autograd. Borrow the arc, not the size.
- `docs/refs/MLX.md`, `MLX_KERNELS.md`, `MLX_REDUCTION.md`: Apple GPU kernel organization at production scale, and the numerical oracle's own semantics.
- `docs/refs/LUMINAL.md`, `LUMINAL_METAL.md`: the compiler-first architecture comtam deliberately rejects, plus the primitive-set idea it adopts.
- `docs/note/AVOID.md`: anti-pattern reference drawn from the author's earlier `legrad` project. Avoid early allocator/backend complexity.
- `refs/metal_example/`: Apple's Metal sample apps, still vendored, mapped by `docs/refs/METAL4.md`.

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

Prefer direct, boring code until repetition or tests prove an abstraction is needed.

## Verification Expectations

For code changes:

- Build with CMake.
- Run the relevant executable or CTest target.
- If Metal execution fails inside a sandbox, state that and run outside the sandbox only with explicit permission.
- Treat printed arrays as debugging, not tests.
- Prefer independent oracles and exact/epsilon comparisons. MLX C running on its CPU stream is sufficient as the numerical oracle; tiny manual CPU oracles are optional when they make a failure easier to understand.
- When the user explicitly says they already built or tested the current revision, do not rerun those commands by default. Record the result as user-verified rather than independently verified, and rerun only when the user asks, evidence conflicts, or a materially risky unresolved issue requires it.

For chapter grading:

- Mention which commands were run.
- Separate "compiled", "smoke ran", and "chapter gate passed".
- Call out missing tests directly.
- Write chapter-level feedback directly in the chapter file's `Agent Feedback / Grading` section, stating status, the evidence that supports it, and the exact commands run.

## Style Notes

- Keep code ASCII unless the file already uses non-ASCII.
- Keep comments sparse and useful.
- Use `rg` for search.
- Use `apply_patch` for manual edits.
- Do not revert user changes unless explicitly asked.
