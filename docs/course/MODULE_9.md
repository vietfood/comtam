## Module 9: Hardening After Correctness

Optimize and harden comtam, but only where a measurement or a failing case
demands it.

## Why This Module Is Last

comtam's rule is strict: no optimization before a failing performance
measurement. Until Module 8 produced a real training run with real timings, you
had nothing to optimize against. Now you do.

This module is where the discipline pays off or collapses. Every change here must
point to a number or a bug, not a feeling that the code "should" be faster or
more general.

## The Hardening Rule

Before any change in this module, write down:

```text
What measurement or failing case forces this?
What is the current number / current failure?
What is the target number / fixed behavior?
How will a test or benchmark confirm it?
```

If you cannot fill those in, do not make the change. Put it in
[`BUG_LEDGER.md`](BUG_LEDGER.md) as deferred instead.

## Assignment 9.1: Build A Tiny Benchmark Harness ⭐⭐

**Task:** Add a small, honest timer around whole ops and whole training steps.

Rules:

- Time end to end (host call to result available), since `Device::submit` blocks
  on `waitUntilCompleted`.
- Warm up before timing (first launch compiles/loads pipelines).
- Report median over several runs, not a single sample.

This is the "simple command timing" stage from [`../AVOID.md`](../AVOID.md), not
GPU counters. Counters come only if this coarse timing is not enough.

**Deliverable:** A benchmark that prints per-op and per-step timings for the
Module 8 model.

## Assignment 9.2: Find The Real Bottleneck ⭐⭐

**Task:** Use the harness to find where time actually goes.

Questions:

1. Is the bottleneck matmul, the reductions, readback, or per-launch overhead
   from blocking on every command?
2. How much time is host-side (allocation, view gather in `to_vector`) versus
   GPU-side?

**Recommended approach:** Measure before you guess. Most people guess matmul and
are sometimes wrong; per-launch synchronization and CPU-side readback gathers are
common surprises.

**Deliverable:** A ranked list of the top two or three costs, with numbers.

## Assignment 9.3: Optimize The Top Bottleneck Only ⭐⭐⭐

**Task:** Make exactly one improvement, targeting the measured top cost.

Candidate optimizations, chosen by what the measurement says:

- **Matmul tiling:** use threadgroup memory to cut global memory traffic. This is
  GPU programming, not framework design, which is why it was deferred this long.
- **Tree reduction:** replace the naive Module 5 reduce with a parallel tree if
  reductions dominate.
- **GPU readback / contiguous-materialize:** replace the CPU gather in
  `to_vector` (Module 2 Design B) if readback dominates.
- **Fewer syncs:** stop blocking on every command if per-launch overhead
  dominates. This is a real ownership change; be careful about lifetime.

Rules:

- One change. Re-run the benchmark. Keep it only if the number improves.
- Re-run the full correctness and gradient test suites. A faster wrong answer is
  worse than a slow right one.

**Why this assignment:** Optimizing the measured bottleneck, one change at a time,
with tests green, is the entire skill. Everything else is folklore.

## Assignment 9.4: Harden The Failure Surface ⭐⭐

**Task:** Turn the sharpest remaining correctness risks into explicit behavior.

Candidates (pick the ones your bug ledger flagged):

1. Division by zero and NaN/Inf handling: define and test the behavior.
2. Shape and dtype validation messages: are they clear at the op boundary?
3. Command buffer error checking after `waitUntilCompleted` (see
   [`../NOTE.md`](../NOTE.md)). Does a kernel failure throw a readable error?
4. Empty and scalar tensors through every op.

**Deliverable:** Tests that pin each chosen behavior, and clear errors where the
operation is genuinely unsupported.

## Assignment 9.5: Decide What To Earn Next ⭐

**Task:** Write a short "what comtam has earned" note.

By now the framework may finally have pressure for things [`../AVOID.md`](../AVOID.md)
told you to postpone. Re-evaluate, honestly:

1. Does repeated code justify an `Allocator`, or just a helper?
2. Does a measurement justify buffer caching?
3. Does a second dtype have a concrete user, or is it speculative?
4. Does any abstraction now pass the test: "what current code became simpler
   because this exists?"

**Deliverable:** A ranked, evidence-backed list of what to build next, and an
equally honest list of what to keep postponing.

## Module 9 Checklist

- [ ] 9.1 Build a warm-up-aware benchmark harness.
- [ ] 9.2 Measure and rank the real bottlenecks.
- [ ] 9.3 Optimize only the top bottleneck; keep tests green.
- [ ] 9.4 Harden the sharpest failure cases with tests.
- [ ] 9.5 Write an evidence-backed "what to earn next" note.

## Exit Criteria

comtam has completed the course when:

1. There is a benchmark harness and at least one optimization justified by it.
2. The full correctness and gradient suites still pass after optimization.
3. Failure cases (bad shapes, kernel errors, edge tensors) have defined,
   tested behavior.
4. Every abstraction in the codebase can answer "what became simpler because of
   it", and you have a written, evidence-backed plan for what comes next.
