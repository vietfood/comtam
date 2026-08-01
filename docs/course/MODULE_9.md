## Module 9: Core Runtime Hardening

Freeze the educational runtime's supported behavior before expanding its product
surface.

## Why This Module Ends The Core Course

Module 8 proves that the happy path trains. A usable core must also define what
happens at its boundaries: empty tensors, scalar tensors, invalid axes, kernel
failures, deterministic reruns, and long-lived execution. This module turns
implicit assumptions into tested contracts.

Performance optimization moves to Module 13. The Module 8 timing baseline may
reveal future work, but this module does not force an optimization merely to
claim progress.

## Module Contract

**Prerequisites:** Module 8 has passed with a recorded training configuration,
accuracy result, lifetime audit, and performance baseline.

**You will produce:**

- a written v0 supported-semantics matrix
- tests for scalar, empty, rank, shape, axis, and numerical edge behavior
- consistent host and Metal error propagation
- deterministic RNG/training replay evidence
- a sustained stress test and explicit ownership counters
- a core-course completion report with deferred work ranked by evidence

**Supported scope:** the exact float32, rank, view, op, autograd, and training
surface implemented through Module 8.

**Not required:** async execution, allocator caching, serialization, Python,
additional backends, broad dtype support, or a performance improvement. Those
belong to the production track and must not be smuggled into this gate.

**Completion evidence:** `docs/solution/MODULE_9.md` must distinguish build,
unit/integration tests, full training, stress results, and any unrun checks.

## Assignment 9.1: Write The Supported-Semantics Matrix ⭐⭐

**Task:** Inventory every public tensor/view/op/autograd/nn entry point and
record its supported inputs and failure behavior.

At minimum, the matrix records:

- dtype support and dtype mismatch behavior
- rank range, including the current GPU metadata rank cap
- scalar and zero-size behavior
- contiguous and non-contiguous input support
- axis normalization and out-of-range behavior
- broadcasting and `keepdim` policy
- whether an op is differentiable
- whether mutation is allowed and how it affects autograd
- synchronization/error boundary

This is a contract extracted from tests and implementation, not an aspirational
feature list. Unsupported cases should be explicit errors, not blank cells.

## Assignment 9.2: Test Scalar, Empty, And Boundary Shapes ⭐⭐⭐

**Task:** Turn each supported edge in the matrix into a test and each unsupported
edge into an error test.

Required categories:

- rank-0 scalar construction, host transfer, elementwise ops, reduction, and
  backward
- zero-length dimensions through views and every op where mathematically defined
- reduction of an empty domain, with an explicit policy per reduction
- rank at the supported maximum and rank above it
- size-one broadcast axes and incompatible dimensions
- negative, normalized, duplicate, and out-of-range axes according to the
  chosen API
- overflow-checked `numel * sizeof(dtype)` allocation sizing

Do not force one universal empty-tensor rule. For example, sum has an identity
while mean of an empty domain is undefined; encode the chosen behavior per op.

## Assignment 9.3: Harden Error Propagation ⭐⭐

**Task:** Ensure invalid user input fails before dispatch and Metal failures
surface after command completion with actionable context.

An error should identify the operation, relevant shapes/dtype/axis, and the
underlying Metal message when present. Tests should assert stable error
categories or meaningful substrings rather than an entire implementation-detail
sentence.

Required cases include invalid matmul shapes, incompatible broadcasts, invalid
reductions, unavailable/missing kernels, buffer byte mismatch, command-buffer
failure where it can be safely induced, and forbidden autograd mutation.

## Assignment 9.4: Prove Reproducibility ⭐⭐

**Task:** Define the relationship between an explicit RNG seed and generated
values, parameter initialization, batching order, and training results.

Run the deterministic Module 7 regression twice from fresh contexts and require
the same initial parameters, batch sequence, and final values within the chosen
policy. Run the Module 8 configuration twice and report accuracy/loss variation;
bitwise identity is not required unless the implementation promises it.

## Assignment 9.5: Run The Core Stress Suite ⭐⭐⭐

**Task:** Combine repeated host transfer, view creation, forward/backward,
optimizer updates, and context construction/destruction into a bounded stress
target.

The test must have a fixed seed and iteration count, check every returned error,
and assert framework-owned object counts return to baseline. Record RSS as
diagnostic evidence, not as the sole pass condition.

Run an AddressSanitizer/UndefinedBehaviorSanitizer host-side configuration where
the toolchain supports it. If Metal or third-party code prevents a sanitizer
configuration, document the exact gap instead of claiming it passed.

## Assignment 9.6: Write The Core Completion Report ⭐

**Task:** Summarize what comtam now supports, what is deliberately unsupported,
and which production-track pressure is evidenced by the training and stress
runs.

Rank deferred work by a concrete reason:

```text
failure or measurement -> proposed module -> acceptance evidence
```

An optimization experiment that does not improve the baseline should be recorded
and rejected. Core completion never requires keeping an unjustified change.

## Module 9 Checklist

- [ ] 9.1 Public supported-semantics matrix matches code and tests.
- [ ] 9.2 Scalar, empty, rank, shape, and axis boundaries are pinned by tests.
- [ ] 9.3 Validation and Metal errors are actionable and tested.
- [ ] 9.4 RNG and training reproducibility policy is demonstrated.
- [ ] 9.5 Bounded stress and available sanitizer checks are recorded.
- [ ] 9.6 Core completion/deferred-work report is evidence-backed.

## Exit Criteria

The core course is complete when:

1. Every public operation has an explicit supported/unsupported contract.
2. Edge tensors and failure paths have deterministic tests rather than implied
   behavior.
3. Full correctness, gradient, training, and stress suites pass in the recorded
   environment.
4. Framework-owned state remains bounded across sustained execution.
5. Reproducibility claims match observed reruns.
6. No optimization or abstraction was added solely because the course gate
   demanded activity.

Completion here means "correct educational eager runtime." Modules 10-15 turn
that core into a narrow production-quality Apple runtime.
