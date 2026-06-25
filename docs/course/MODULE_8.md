# Module 8: Hardening Without Bloat

Only harden what the working framework has proven it needs.

## Add These First

- better error messages
- debug tensor names
- op tracing
- simple command timing
- leak counters
- a small test matrix in CI

## Add These Only With Evidence

- caching allocator
- private Metal buffers
- async execution
- kernel specialization
- batched matmul
- half precision
- serialization

Evidence means a failing benchmark, correctness limitation, or painful repeated
code path. "Real frameworks have this" is not evidence.

## Assignment 8.1: Op Trace

Add an optional trace like:

```text
add metal f32 shape=(32, 64)
matmul metal f32 lhs=(32, 128) rhs=(128, 64) out=(32, 64)
sum metal f32 shape=(32, 64) axis=all
```

## Assignment 8.2: Timing

Measure command buffer execution time for individual ops.

Do not redesign the runtime around profiling.

## Assignment 8.3: Hardening Ledger

Create a ledger of possible improvements:

```text
Issue:
Evidence:
Smallest fix:
Risk:
Decision:
```

## Completion Gate

The course is complete when:

- all tests pass
- linear regression trains
- XOR trains
- op traces are readable
- no hardening feature has entered without evidence

