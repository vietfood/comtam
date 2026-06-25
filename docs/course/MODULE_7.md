## Module 7: nn Modules And SGD

Wrap the tensor and autograd machinery in the smallest layer that lets you build
and train a network.

## Why This Module Exists

You now have differentiable ops. A network is just a tidy way to organize
parameters and the forward function that uses them, plus an optimizer that walks
those parameters downhill.

This module is intentionally thin. The risk here is building a Keras. Resist it.
The whole point is to prove the stack from Module 1 through Module 6 composes into
something trainable, not to design a layer zoo.

## Mental Model

Three small concepts, from [`../ARCHITECTURE.md`](../ARCHITECTURE.md):

```text
Parameter   a Tensor with requires_grad = true that an optimizer updates
Module      owns parameters, defines forward(); can contain submodules
Optimizer   given parameters, applies an update rule from their grads
```

A training step is always the same shape:

```text
zero grads
forward:   y = model(x)
loss:      l = loss_fn(y, target)   // scalar
backward:  l.backward()
step:      optimizer updates each parameter from its grad
```

If that loop runs and the loss goes down, the whole framework works.

## Assignment 7.1: Implement `Parameter` And Parameter Collection ⭐⭐

**Task:** Add a `Parameter` (a `Tensor` with `requires_grad = true`) and a way
for a `Module` to enumerate all its parameters, including those in submodules.

Questions:

1. How does a module register parameters: explicit registration, or reflection
   over members? In C++ there is no reflection, so explicit registration is the
   honest choice. What does that API look like?
2. Who owns the parameter tensors: the module? For how long?
3. How do nested modules expose a flat parameter list to the optimizer?

**Recommended first choice:** Explicit registration into a `std::vector` of
parameter pointers per module, with a `parameters()` method that recurses into
registered submodules. Boring and obvious beats clever.

**Test:** A module with two parameters and one submodule returns all three from
`parameters()`.

## Assignment 7.2: Implement `Linear` ⭐⭐

**Task:** Add `nn::Linear(in, out)` computing `y = x @ W + b`.

Requirements:

- `W` and `b` are `Parameter`s, initialized with the RNG in
  `comtam/utils/rng.h` (a small, reasonable init; full init schemes can wait).
- forward uses your Module 5 matmul and your Module 5 broadcasting (the bias is
  broadcast over the batch).

Questions:

1. What are the shapes of `W` and `b`, and does the bias add rely on
   broadcasting being correct (Module 6)?
2. Does backward through `Linear` need any new rule, or does it fall out of
   matmul + add gradients you already have?

**Test:** Forward of `Linear` against a CPU oracle, and a numerical gradient
check on `W` and `b` through a tiny loss.

## Assignment 7.3: Implement A Loss And An Activation ⭐⭐

**Task:** Add one activation and one loss, both differentiable.

- activation: `relu` (forward `max(x, 0)`, backward gradient masked where `x<=0`)
- loss: mean squared error, returning a scalar

Questions:

1. `relu` needs a new elementwise kernel and a new backward rule. Does it fit the
   Module 3 dispatch pattern cleanly, or does the binary-only `Command` need to
   grow a unary form?
2. Does MSE reduce to ops you already have (sub, mul/square, mean)?

**Recommended first choice:** Build MSE from existing ops so it needs no new
backward rule. Add `relu` as the one genuinely new differentiable kernel, and use
it to test that adding a unary op is clean.

**Test:** Numerical gradient check through `relu` and through MSE.

## Assignment 7.4: Implement SGD ⭐⭐

**Task:** Add an `SGD` optimizer.

```text
for each parameter p:
    p.data = p.data - lr * p.grad
zero_grad(): set each p.grad to zero (or null) before the next step
```

Questions:

1. Should the parameter update run on the GPU (a kernel) or on the host? For a
   first version, which is simpler to get correct?
2. Where does `zero_grad` belong in the training step, and what bug appears if
   you forget it? (Hint: Module 6 gradients accumulate.)

**Recommended first choice:** Implement the update with existing tensor ops on
the GPU so it exercises the same path as everything else. Make `zero_grad`
explicit; do not hide it inside `step`.

**Test:** One SGD step on a single parameter with a known gradient moves it by
exactly `-lr * grad`.

## Assignment 7.5: The Smallest Training Loop ⭐⭐⭐

**Task:** Fit a single `Linear` layer to a linear function with known weights,
for example `y = 2x + 1` plus small noise.

Steps:

1. Generate data with the RNG utility.
2. Run the standard training step in a loop.
3. Assert the loss decreases and the learned `W`, `b` approach the true values.

**Why this assignment:** This is the first end-to-end proof. If a one-layer
regression converges, parameters, forward, autograd, optimizer, and data movement
are coherent. This is the gate that retroactively validates Modules 1 through 6.

## Module 7 Checklist

- [ ] 7.1 Implement `Parameter` and recursive `parameters()`.
- [ ] 7.2 Implement `Linear` with forward and gradient tests.
- [ ] 7.3 Implement `relu` and MSE loss with gradient checks.
- [ ] 7.4 Implement SGD with explicit `zero_grad`.
- [ ] 7.5 Train one `Linear` layer to convergence on a known function.

## Exit Criteria

You are ready for Module 8 when:

1. A `Module` exposes a flat parameter list, including submodules.
2. `Linear`, `relu`, and MSE all pass forward and gradient tests.
3. SGD updates parameters correctly, with explicit `zero_grad`.
4. A one-layer regression converges to known weights, verified by assertion.
