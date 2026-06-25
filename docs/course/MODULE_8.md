## Module 8: Small Training Examples

Prove comtam can learn a real task end to end.

## Why This Module Exists

Module 7 trained a single linear layer. That proves the plumbing connects. This
module proves the plumbing is strong enough for a real, multi-layer, multi-class
problem with a real dataset.

The classic gate is a small MLP on MNIST. It is the smallest task that exercises
multiple layers, a real loss, batching, and many optimizer steps in a row. If
comtam trains it to a believable accuracy, the eager stack is genuinely working.

## Mental Model

Nothing new conceptually. The same training step, scaled up:

```text
for each epoch:
  for each batch (x, target):
    optimizer.zero_grad()
    logits = model(x)
    loss = cross_entropy(logits, target)
    loss.backward()
    optimizer.step()
```

The new pressures are practical: data loading, batching, numerical stability of
the loss, and running thousands of GPU commands without leaking.

## Assignment 8.1: Load MNIST Without A Dependency Zoo ⭐⭐

**Task:** Load MNIST into host float arrays, then into tensors.

Questions:

1. Where does the data come from, and how do you parse the IDX format into flat
   `float` vectors without pulling in a heavy library?
2. How do you normalize inputs, and why does normalization matter for
   convergence?
3. How do you batch: re-upload each batch to a fresh tensor, or reuse buffers?

**Recommended first choice:** Parse the raw IDX files into `std::vector<float>`
yourself. Keep IO out of the framework core; this is example code, not a comtam
subsystem (see [`../AVOID.md`](../AVOID.md) on image/audio IO).

## Assignment 8.2: Implement Softmax Cross-Entropy ⭐⭐⭐

**Task:** Add a numerically stable cross-entropy loss over logits.

Requirements:

- subtract the row max before exponentiating (stability)
- combine softmax and the log into one loss to avoid `log(0)`
- return a scalar, and provide its backward rule

Questions:

1. Why does the naive `log(softmax(x))` blow up, and how does the max-subtraction
   trick fix it?
2. Can you express the backward as `softmax(logits) - one_hot(target)`, scaled by
   batch size? Derive it and confirm with a numerical gradient check.

**Test:** Numerical gradient check on a small logits/target pair, and a stability
check with large-magnitude logits.

## Assignment 8.3: Build And Train The MLP ⭐⭐⭐

**Task:** Train a 2-layer MLP (`Linear -> relu -> Linear`) on MNIST.

Steps:

1. Assemble the model from Module 7 components.
2. Write the epoch/batch loop with the standard training step.
3. Track training loss and validation accuracy.
4. Reach a believable accuracy (>90% is a reasonable first target for an MLP).

Reporting rule (the module gate):

```text
compiled    -> built
smoke ran   -> one batch ran forward+backward without crashing
gate passed -> trained for real epochs and reached the accuracy target
```

Do not claim the gate on a smoke run. A loss that prints is not a loss that
converged.

**Why this assignment:** This is the capstone of correctness. Multiple layers,
real gradients, real data, and many steps either compose into learning or they
do not.

## Assignment 8.4: Watch For Lifetime And Leak Bugs ⭐⭐

**Task:** Run a full training run and watch memory and stability.

Questions:

1. Does memory grow per iteration? If so, revisit the autorelease pool policy
   from Assignment 3.5 (a per-iteration pool).
2. Does the autograd graph from one step get freed before the next, or does it
   pin every intermediate tensor forever?
3. Are gradients zeroed each step, or are they silently accumulating across
   steps?

**Why this assignment:** Bugs that are invisible at one step become obvious over
ten thousand. A training run is the best leak detector you have before Module 9.

## Module 8 Checklist

- [ ] 8.1 Load and normalize MNIST into tensors.
- [ ] 8.2 Implement stable softmax cross-entropy with a gradient check.
- [ ] 8.3 Train a 2-layer MLP to a believable accuracy.
- [ ] 8.4 Run a full training pass and audit memory and graph lifetime.

## Exit Criteria

You are ready for Module 9 when:

1. A 2-layer MLP trains on MNIST and reaches its accuracy target across real
   epochs, not a smoke run.
2. Cross-entropy is numerically stable and gradient-checked.
3. A full training run does not leak memory or accumulate stale gradients.
4. You have at least one measured number (time per epoch, or per step) to make
   Module 9's optimization decisions evidence-based.
