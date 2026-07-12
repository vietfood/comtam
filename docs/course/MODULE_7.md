## Module 7: Neural-Network Modules And SGD

Wrap the tensor and autograd machinery in the smallest abstraction that can
train a model without changing parameter identity or recording optimizer work.

## Why This Module Exists

A network is an ownership structure around parameters plus a forward function.
An optimizer is a stateful procedure that updates those parameters from leaf
gradients. The important lesson is not a large layer library; it is proving that
tensor identity, autograd, parameter registration, and mutation compose safely.

```text
optimizer.zero_grad()
prediction = model.forward(input)
loss = loss_fn(prediction, target)
loss.backward()
optimizer.step()          // under no_grad, same parameters remain registered
```

## Module Contract

**Prerequisites:** Module 6 has passed, including stable leaf identity, scoped
`no_grad`, graph release, and explicit gradient clearing.

**You will produce:**

- `nn::parameter` and a non-ambiguous registration/ownership policy
- `nn::module` with recursive named parameter enumeration
- deterministic `nn::linear`, `relu`, and mean squared error
- SGD whose update preserves registered parameter identity
- a deterministic one-layer regression convergence test

**Supported scope:** float32, explicit context ownership, dense layers, one
activation, one loss, and plain SGD without momentum or weight decay.

**Not required:** reflection, a layer zoo, mixed precision, train/eval mode,
optimizer checkpointing, parameter groups, schedulers, or automatic batching.

**Completion evidence:** record the public API chosen, exact initialization
seed, convergence thresholds, build command, and CTest command in
`docs/solution/MODULE_7.md`.

## Parameter Identity Rule

Registration identifies a parameter, not the temporary value produced by an
update expression. If a module registers parameter `weight`, then after every
optimizer step:

- `parameters()` still returns the same parameter identity
- existing handles observe the new values
- the parameter remains a leaf requiring gradients
- no `GradNode` describes the optimizer update
- its old gradient is cleared only when `zero_grad()` is called

An implementation equivalent to `p = p - lr*p.grad` is wrong if it replaces the
registered handle or records a new graph. Compute an update under `no_grad` and
copy it into the existing parameter storage, or provide a narrow in-place
parameter update that enforces the same rules.

## Assignment 7.1: Implement Parameter Registration ⭐⭐⭐

**Task:** Add `nn::parameter` and `nn::module` with explicit registration.
C++ has no suitable member reflection here, so registration must be visible in
the constructor or through named registration calls.

The minimum API must support the equivalent of:

```cpp
register_parameter("weight", weight);
register_module("child", child);
named_parameters(); // recursive names such as "child.weight"
parameters();       // stable handles for optimizers
```

Choose and document whether modules are non-copyable, movable, or shared-owned.
Raw parameter pointers are acceptable only when the module lifetime and move
policy make them stable. Duplicate names and registration of the same parameter
under conflicting names must fail clearly.

**Tests:**

- a module with two parameters and a child with one parameter returns three
  distinct identities with deterministic names
- repeated enumeration returns the same identities and order
- duplicate registration is rejected
- moving/copying a module either works according to the documented policy or is
  rejected at compile time

## Assignment 7.2: Implement Deterministic Linear ⭐⭐

**Task:** Add `nn::linear(in_features, out_features)` computing:

```text
y = x @ W + b
W shape = (in_features, out_features)
b shape = (out_features)
x shape = (batch, in_features)
y shape = (batch, out_features)
```

Use the existing matmul and broadcasting paths. Initialization must accept an
explicit seed or RNG object so the test is reproducible. Pick one simple
documented scheme; a small uniform range is sufficient.

**Tests:** compare forward values to a manual CPU calculation with fixed weights,
then gradient-check `x`, `W`, and `b` through a scalar loss. Also reject an input
whose final dimension does not equal `in_features`.

## Assignment 7.3: Implement ReLU And MSE ⭐⭐

**Task:** Add one new unary differentiable op, `relu`, and compose mean squared
error from existing tensor ops:

```text
relu(x) = max(x, 0)
error = prediction - target
mse(prediction, target) = sum(error * error) / error.numel()
```

Define `relu'(0)=0` and test that exact boundary. MSE uses Module 5's full `sum`
plus scalar broadcasting; it does not require a separate full-mean API. Reject
empty input because division by `numel()==0` is undefined. MSE must return a
rank-0 scalar and follow Module 5 broadcasting policy only if you intentionally
allow target broadcasting; otherwise require equal shapes and say so.

**Tests:** forward oracle and numerical gradient tests for negative, zero, and
positive ReLU inputs; MSE forward/gradient checks; shape mismatch rejection.

## Assignment 7.4: Implement SGD Without Recording Updates ⭐⭐⭐

**Task:** Add an optimizer constructed from stable parameter handles and a
positive finite learning rate.

`step()` must:

1. reject or skip a missing gradient according to one documented policy
2. enter `no_grad`
3. update the existing parameter identity from `p - lr*p.grad`
4. leave gradient clearing to `zero_grad()`

Use `zero_grad()` to set each leaf gradient to absent/null for the first version.
This makes forgotten backward calls distinguishable from legitimate zero
gradients. Do not hide `zero_grad()` inside `step()`.

**Tests:**

- one known SGD step changes values by exactly `-lr*grad` within float tolerance
- parameter identity and registration names are unchanged after the step
- the updated parameter is still a leaf and has no optimizer `GradNode`
- two backward calls without `zero_grad()` accumulate, while clearing between
  steps prevents stale accumulation
- invalid learning rates fail clearly

## Assignment 7.5: Fit A Known Linear Function ⭐⭐⭐

**Task:** Fit a single linear layer to deterministic data generated from
`y = 2x + 1`. Do not add noise to the required convergence test; noise can be a
separate experiment.

Use a fixed seed, fixed dataset, explicit learning rate, and a maximum iteration
count. Assert all three outcomes:

- final loss is below a stated threshold
- learned weight and bias are within stated tolerances of 2 and 1
- loss decreases substantially from its initial value

Printing a decreasing loss is a debugging aid, not the test. The convergence
run must be a CTest target with bounded runtime.

## Assignment 7.6: Prove Parameter And Graph Lifetime ⭐⭐

**Task:** Run repeated training steps and prove that parameter identities remain
stable while per-step graphs are released. Reuse the deterministic live-state
instrumentation from Module 6.

The test should retain one external parameter handle, train for many steps, and
confirm that the handle observes updates without keeping old graphs alive.

## Module 7 Checklist

- [ ] 7.1 Explicit named parameter/submodule registration with stable ownership.
- [ ] 7.2 Deterministic `linear` forward and gradient tests.
- [ ] 7.3 ReLU and MSE forward/gradient tests with defined edge behavior.
- [ ] 7.4 SGD updates under `no_grad` without replacing parameter identity.
- [ ] 7.5 Deterministic linear regression passes convergence assertions.
- [ ] 7.6 Repeated training preserves parameters and releases graphs.

## Exit Criteria

You are ready for Module 8 when:

1. Recursive named parameter enumeration is deterministic and identity-stable.
2. `linear`, ReLU, and MSE pass forward and numerical gradient tests.
3. SGD changes existing leaf parameters without recording an update graph.
4. Gradient clearing and accumulation have explicit, tested behavior.
5. The deterministic regression test converges within its fixed iteration and
   accuracy bounds.
6. Repeated steps do not retain previous autograd graphs.
