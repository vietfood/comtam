## Module 8: End-To-End Training

Train a small multi-layer classifier and turn one-step correctness into sustained
runtime evidence.

## Why This Module Exists

Module 7 proves that one layer can learn a deterministic function. A real
mini-batch classifier adds multiple layers, a numerically sensitive loss, data
loading, repeated GPU submission, and validation metrics. These pressures expose
graph leaks, unstable math, stale gradients, and ownership errors that a single
forward/backward test cannot reveal.

## Module Contract

**Prerequisites:** Module 7 has passed, including stable parameter identity,
graph release, deterministic initialization, and SGD.

**You will produce:**

- a small MNIST IDX loader kept outside the framework core
- an explicit target representation compatible with the dtype policy
- stable softmax cross-entropy with forward and gradient oracles
- a deterministic two-layer MLP training executable/test
- measurable convergence, validation accuracy, timing, and lifetime evidence

**Supported scope:** MNIST, host-side batching, float32 tensors, a two-layer MLP,
ReLU, softmax cross-entropy, and SGD.

**Not required:** a generic data-loader subsystem, image IO in `comtam_lib`, GPU
data augmentation, convolution, mixed precision, or distributed training.

**Completion evidence:** record dataset identity/checksum or fixture source,
seed, batch size, model dimensions, learning rate, epochs, final accuracy, timing,
and memory/lifetime results in `docs/archive/course-v1/solution/MODULE_8.md`.

## Target Representation Decision

The framework is float32-first, while class-index targets normally use an integer
dtype. Do not let MNIST silently introduce an incomplete dtype system.

The recommended core-course policy is float32 one-hot targets of shape
`(batch, classes)`. Cross-entropy then accepts:

```text
logits:  float32 (batch, classes)
targets: float32 (batch, classes), one-hot rows
result:  float32 scalar
```

Validate that shapes match and each target row is a valid one-hot vector within
the chosen tolerance. Accuracy may copy logits to the host and compute `argmax`
there.

You may instead introduce integer class-index targets, but then the assignment
also includes defining that dtype's storage size, host transfer, validation, and
kernel contract. Adding an integer enum without those semantics does not count.

## Assignment 8.1: Parse A Reproducible MNIST Fixture ⭐⭐

**Task:** Implement IDX image/label parsing in example or test-support code,
never in the tensor runtime.

The parser must:

- validate IDX magic numbers and big-endian dimensions
- reject truncated files and inconsistent image/label counts
- normalize pixels to `[0,1]` float32
- convert labels to the chosen target representation
- return deterministic batches without hidden shuffling

Do not make the required unit tests depend on a network download. Check in a
small legal fixture or construct tiny IDX byte fixtures in the test. The full
training executable can accept an explicit dataset directory.

**Tests:** valid miniature image/label files, wrong magic, truncation, mismatched
counts, normalization endpoints, and target conversion.

## Assignment 8.2: Implement Stable Softmax Cross-Entropy ⭐⭐⭐

**Task:** Add one public differentiable operation,
`softmax_cross_entropy(logits, targets)`, backed by a dedicated forward path and
local backward rule. Do not add public `max`, `exp`, or `log` tensor operations
as side effects of this assignment; those belong in an operator-coverage project
with their own semantic and gradient contracts.

The forward kernel or internal implementation uses log-sum-exp stability per
row:

```text
row_max = maximum class logit in the row
shifted = logits - row_max
logsumexp = log(sum(exp(shifted)))
row_loss = -sum(targets * (shifted - logsumexp))
loss = sum(row_loss) / batch_size
```

`row_max`, exponentiation, logarithm, and softmax are private implementation
details of this op in the core course. They do not enter the public op enum/API
unless a later module deliberately specifies and tests them.

For one-hot targets, the logits gradient is:

```text
(softmax(logits) - targets) / batch_size
```

Save only what backward needs. Define behavior for batch size zero as an error;
do not return a NaN loss silently.

**Tests:**

- forward comparison to a double-precision CPU oracle
- numerical gradient check on a small logits matrix
- invariance to adding the same constant to every class in a row
- finite loss and gradients for logits near `+1000` and `-1000`
- invalid target shape/content and empty batch rejection

## Assignment 8.3: Build The MLP And Training Loop ⭐⭐⭐

**Task:** Train `linear -> relu -> linear` using explicit mini-batches:

```text
for each epoch:
  for each batch:
    optimizer.zero_grad()
    logits = model.forward(x)
    loss = softmax_cross_entropy(logits, target)
    loss.backward()
    optimizer.step()
```

Use a fixed seed and record the full hyperparameter configuration. Start with a
small subset or one batch for debugging, but that does not satisfy the gate.

The required report distinguishes:

```text
compiled    -> executable and tests built
smoke ran   -> one batch completed forward/backward/update
gate passed -> full configured run reached the accuracy and stability gates
```

**Training gate:** reach at least 90% validation accuracy on MNIST with the
documented configuration. Because GPU and floating-point ordering can vary,
choose a deterministic seed and a threshold with reasonable margin rather than
asserting one exact loss trajectory.

## Assignment 8.4: Measure Sustained Lifetime Behavior ⭐⭐⭐

**Task:** Run at least 10,000 training steps on one fixed small model and a fixed
deterministic batch sequence, or the entire configured training run if longer,
and separate deterministic ownership checks from OS memory observations. Record
the seed, model shape, batch contents/order, optimizer configuration, and exact
iteration count so the stress run is reproducible.

Required evidence:

- live autograd nodes/states return to the post-step baseline
- previous batch tensors become unreachable
- parameter identities remain constant
- gradients are absent immediately after `zero_grad()`
- command-buffer errors are checked every step
- autorelease pools are drained at the documented boundary

Also sample resident memory after a stated warm-up and report the observed band.
Allocator and driver caching can make RSS non-zero or non-monotonic, so RSS is a
diagnostic report rather than a pass condition; stable framework-owned counts are
the deterministic gate.

## Assignment 8.5: Record A Performance Baseline ⭐⭐

**Task:** Record median batch time and epoch time after warm-up, including the
synchronization boundary used for timing. This is a baseline for Module 13, not
permission to optimize now.

Report hardware, macOS version, build type, batch size, model shape, warm-up
count, sample count, median, and a dispersion measure. A single timing sample is
not evidence.

## Module 8 Checklist

- [ ] 8.1 Validated IDX parser with offline fixtures.
- [ ] 8.2 Explicit target contract and stable cross-entropy gradient.
- [ ] 8.3 Full MLP run reaches the documented accuracy threshold.
- [ ] 8.4 Repeated-step ownership and memory behavior are measured.
- [ ] 8.5 Reproducible batch/epoch performance baseline is recorded.

## Exit Criteria

You are ready for Module 9 when:

1. Cross-entropy matches a CPU oracle, passes its numerical gradient check, and
   remains finite on large logits.
2. A real multi-epoch MNIST run reaches at least 90% validation accuracy.
3. The target representation and dtype boundary are explicit and tested.
4. The sustained run releases per-step graphs and does not show unbounded
   framework-owned state.
5. The complete training configuration and performance baseline are recorded so
   another run can reproduce the result.
