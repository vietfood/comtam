## Module 6: Reverse-Mode Autograd

Build a dynamic backward graph whose ownership and failure behavior are as
explicit as its gradient formulas.

## Why This Module Comes After Forward Correctness

Autograd is a second system layered on the forward tensor runtime. Each eager op
runs immediately, then records enough information for a later backward pass.
If forward shapes, broadcasting, or views are wrong, backward will compose those
errors into gradients that may look plausible while training the wrong model.

The minimum execution model is:

```text
forward op
  -> compute output now
  -> if recording is enabled and an input requires gradients,
     attach a GradNode to the output

backward(scalar loss)
  -> topologically order reachable GradNodes
  -> seed dloss/dloss = 1
  -> run local rules in reverse order
  -> add contributions into leaf gradients
  -> release saved graph state
```

## Module Contract

**Prerequisites:** Module 5 has passed, including broadcasted binary ops,
reductions, and strided-input matmul. Every forward op covered here already has
a forward correctness test.

**You will produce:**

- stable tensor/autograd identity across ordinary tensor-handle copies
- `requires_grad`, leaf gradients, and a per-result `GradNode`
- a scoped recording guard (`no_grad`) and `detach`
- reverse topological traversal with additive gradient accumulation
- backward rules for the supported elementwise, movement, reduction, and
  matmul ops
- a reusable central-difference gradient checker under CTest
- tests proving graph lifetime and repeated-use behavior

**Supported scope:** float32 tensors, one explicit `core::context`, scalar-root
backward, one backward pass per recorded graph, and the differentiable ops named
in this module.

**Not required:** higher-order gradients, `retain_graph`, custom user-defined
backward functions, multithreaded backward, distributed autograd, or in-place
mutation of tensors saved for backward.

**Completion evidence:** record the exact build and CTest commands, the gradient
checker tolerances, and a short hand trace of one branched graph in
`docs/solution/MODULE_6.md`.

## Required Semantics Before Coding Rules

These definitions are part of the API contract:

- A **leaf** is a user-created tensor with `requires_grad=true` and no producing
  `GradNode`. Only leaves accumulate persistent `.grad` values in this module.
- A **non-leaf** is a differentiable op result. It carries a `GradNode`, but its
  temporary upstream gradient may be released after backward.
- A tensor-handle copy refers to the same autograd identity. Copying a tensor
  value must not create a second independent `.grad` slot.
- A movement op creates a new logical result and therefore a new autograd
  identity, even when it shares the same storage.
- `detach()` returns a tensor sharing storage and view metadata while starting a
  new identity with `requires_grad=false` and no `GradNode`.
- `no_grad` disables node recording within a lexical scope and restores the
  previous state on exit, including exceptional exit.
- Calling `backward()` on a non-scalar root is an error in this module.
- Backward consumes the graph. A second call on that graph must fail clearly;
  `retain_graph` is deliberately deferred.

A small shared internal autograd state is the recommended implementation. It
can hold `requires_grad`, the leaf gradient, and the producing node while the
existing tensor header continues to own dtype/view/storage interpretation. Do
not put a `tensor` directly inside itself; use handles such as `shared_ptr` where
recursive ownership is required, and draw the ownership graph before coding.

## Assignment 6.1: Design Tensor And GradNode Identity ⭐⭐⭐

**Task:** Write the design note first, then implement the minimum state needed
to satisfy the semantics above.

Your note must answer:

1. Which object defines tensor identity, and what happens when a tensor handle
   is copied?
2. Who owns a `GradNode`, its input tensor handles, and tensors saved for
   backward?
3. Why does the ownership graph have no strong-reference cycle?
4. When are saved tensors released?
5. How is recording state associated with an explicit context instead of a
   process-wide hidden singleton?

At minimum, a node needs an op/rule identifier, its differentiable inputs,
saved values or metadata required by the rule, and a way to return one gradient
contribution per input. Do not use raw pointers whose lifetime depends on the
caller keeping local tensor variables alive.

**Tests:**

- copying a leaf handle and using either copy observes the same accumulated
  gradient
- a temporary input remains alive until its backward rule has run
- a view result has a distinct autograd identity while sharing storage
- destroying the scalar loss after backward releases the reachable graph

## Assignment 6.2: Implement Recording Modes And Root Backward ⭐⭐⭐

**Task:** Implement `requires_grad` propagation, `detach`, scoped `no_grad`,
topological ordering, scalar seeding, and graph consumption before adding all
local rules.

The propagation rule is precise: an op result requires gradients when recording
is enabled and at least one differentiable input requires gradients. Constants
and inputs with `requires_grad=false` do not receive gradient slots.

Use a visited set during topological traversal because a dynamic graph is a DAG,
not necessarily a tree. For `z = x*x + x`, the leaf `x` is reached through
multiple edges but its producing state must be visited once and its three local
contributions must be added.

**Tests:**

- `backward()` rejects a non-scalar root and a root that does not require grad
- recording is disabled inside `no_grad` and restored afterward
- `detach` shares forward values but blocks gradient flow
- a diamond/branched graph is topologically processed once per node
- a second backward through a consumed graph throws a readable error

## Assignment 6.3: Build The Numerical Gradient Checker ⭐⭐⭐

**Task:** Add a reusable test helper that compares autograd with central finite
differences for a scalar-valued function:

```text
grad_numeric[i] = (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)
```

Use small deterministic tensors. Start with `eps=1e-3` for float32 and document
both absolute and relative tolerances; do not silently loosen them per failing
operator. The numerical evaluations must run under `no_grad`, otherwise the
checker builds graphs that it never needs.

The helper must report the failing logical index, analytic value, numerical
value, absolute error, and allowed tolerance. A checker that only returns false
is too difficult to debug.

**Self-test:** Run the checker against a simple CPU-understood function such as
`sum(x*x)` and intentionally perturb one expected gradient to prove it catches
an error.

## Assignment 6.4: Elementwise Rules And Accumulation ⭐⭐

**Task:** Add backward rules for `add`, `sub`, `mul`, and `div`:

```text
add:  da = dout,             db = dout
sub:  da = dout,             db = -dout
mul:  da = dout * b,         db = dout * a
div:  da = dout / b,         db = -dout * a / (b*b)
```

Build gradients from existing forward tensor ops while recording is disabled.
Every contribution is added to any existing contribution; never overwrite a
gradient merely because it is the first edge encountered during traversal.

**Tests:** gradient-check every op on deterministic small inputs, keep division
denominators away from zero, and include `sum(x*x + x)` to prove repeated-use
accumulation.

## Assignment 6.5: Movement And Broadcasting Rules ⭐⭐⭐

**Task:** Implement these inverse mappings:

| Forward op | Backward mapping |
| --- | --- |
| `reshape` | reshape to the input shape |
| `permute` | apply the inverse permutation |
| `expand` | sum over introduced and stride-zero axes |
| `shrink` | scatter into zeros at the original slice |

Broadcasted binary ops use the same `sum_to_shape` operation as `expand`
backward. Given an upstream gradient and an original input shape,
`sum_to_shape` must reduce leading axes introduced by rank expansion and axes
where the original size was one, then restore the original rank.

**Required cases:**

- `(3,)` broadcast into `(4,3)`
- `(2,1,4)` broadcast into `(2,3,4)`
- transpose and general permute
- a non-zero-offset shrink
- contiguous reshape

Each case needs a numerical gradient check and an exact gradient-shape check.

## Assignment 6.6: Reduction And Matmul Rules ⭐⭐⭐

**Task:** Add gradients for full/axis `sum`, axis `mean`, and 2-D `matmul`:

```text
sum(all):       expand dout to input shape
sum(axis):      reinsert the reduced axis when needed, then expand
mean(axis):     sum backward scaled by 1/reduced_count
C = A @ B:      dA = dout @ transpose(B), dB = transpose(A) @ dout
```

The rule must honor Module 5's chosen `keepdim` policy. Do not infer the reduced
axis later from ambiguous shapes; save the normalized axis and input shape in
the node.

**Tests:** gradient-check full sum, both axes of a 2-D sum and axis mean, a small
matmul, and matmul with a transposed input view.

## Assignment 6.7: Graph Lifetime And Mutation Boundary ⭐⭐⭐

**Task:** Prove graph state does not leak across training steps and define the
mutation boundary used by Module 7.

For the core course, mutation of a tensor saved for backward between forward and
backward is unsupported. Either reject it with a lightweight storage version
check or ensure the public mutation path cannot perform it. Document the exact
policy; silently computing a gradient from changed values is not acceptable.

After backward, saved tensors and non-leaf upstream gradients must be released.
Leaf gradients remain until explicitly cleared. Parameter updates in Module 7
will occur only after backward, under `no_grad`, and must preserve parameter
identity.

**Stress test:** execute a small forward/backward graph repeatedly, clear leaf
gradients each iteration, and assert framework-owned live node/state counts
return to a stable baseline. An OS memory graph may supplement this test but
does not replace deterministic ownership evidence.

## Assignment 6.8: Read Magnetron Autograd ⭐⭐

Only after your own design works, trace Magnetron's tensor autograd state,
topological traversal, and gradient rules in `refs/magnetron`.

Write a short note containing:

- one ownership or traversal idea comtam should copy
- one feature comtam should postpone
- one semantic difference caused by comtam's explicit context and Metal-only
  scope

## Module 6 Checklist

- [ ] 6.1 Stable tensor/autograd identity and documented ownership.
- [ ] 6.2 Recording modes, detach, traversal, scalar backward, graph consumption.
- [ ] 6.3 Diagnostic numerical gradient checker under CTest.
- [ ] 6.4 Elementwise rules with additive accumulation.
- [ ] 6.5 Movement and broadcast reduction rules.
- [ ] 6.6 Reduction and matmul rules.
- [ ] 6.7 Graph lifetime and mutation boundary tests.
- [ ] 6.8 Magnetron copy/postpone comparison note.

## Exit Criteria

You are ready for Module 7 when:

1. Every supported differentiable op has a passing numerical gradient test and
   an exact gradient-shape assertion.
2. Copies, branches, broadcasting, and repeated tensor use accumulate into the
   correct leaf identity.
3. `detach` and scoped `no_grad` have tested behavior.
4. Backward consumes and releases its graph, while leaf gradients persist until
   explicitly cleared.
5. Unsupported roots, repeated backward, and forbidden mutation fail clearly.
6. You can trace one branched scalar loss by hand and match the implementation's
   topological order and accumulated values.
