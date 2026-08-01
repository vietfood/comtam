## Module 6: Reverse-Mode Autograd

Build a dynamic backward graph whose ownership and failure behavior are as
explicit as its gradient formulas, using the semantic primitive surface fixed
and forward-tested by mandatory Module 5A.

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

## The Primitive Surface Pays Off Here

Module 5A fixed the differentiable arithmetic surface at `add`, `mul`, `neg`,
`recip`, `sum`, and `matmul`, with movement operations as their own semantic
nodes. `sub`, `div`, and `mean` are tensor-layer compositions. That decision
was partly made for this module:

```text
composed op forward:   sub(a, b) runs neg(b) then add(a, .)
                       (assume both a and b require gradients here)
tape records:          two primitive GradNodes, exactly as if the user
                       had written the composition by hand
backward:              the neg and add rules do all the work;
                       no sub rule exists or is needed
```

The complete set of hand-written backward rules in this module is therefore:

| Forward | Backward rule |
| --- | --- |
| `add` | `da = dout`, `db = dout` (with broadcast reduction) |
| `mul` | `da = dout * b`, `db = dout * a` (with broadcast reduction) |
| `neg` | `dx = neg(dout)` |
| `recip`, saving `y = recip(x)` | `dx = neg(dout * y * y)` |
| `sum` | expand `dout` back to the input shape |
| `matmul` | `dA = dout @ B^T`, `dB = A^T @ dout` |
| movement ops | inverse view mappings (Assignment 6.6) |

The reciprocal rule uses the saved output instead of `x*x`: squaring a large
finite `x` can overflow and incorrectly collapse the derivative to zero. Keep
reciprocal gradient-check inputs finite and away from zero; Module 5A already
owns exceptional forward semantics.

`sub`, `div`, and `mean` appear in this table nowhere. They gradient-check
through the tape the day it exists, and your tests must prove that explicitly -
an untested assumption of "composition handles it" is exactly how silent
gradient bugs are born. Luminal's autograd works on the same principle,
statically: its training crate derives local gradients for the small primitive
set only and wires the backward graph together ahead of time
([`../refs/LUMINAL.md`](../refs/LUMINAL.md)). comtam gets the same economy
dynamically, one recorded node at a time.

## Module Contract

**Prerequisites:** Module 5 and mandatory Module 5A have passed. The runtime has
broadcasted `add`/`mul`, strided unary `neg`/`recip`, semantic `sum`,
strided-input `matmul`, and composed `sub`/`div`/full-and-axis `mean`. Every
forward primitive covered here already has an independent correctness test.

**You will produce:**

- stable tensor/autograd identity across ordinary tensor-handle copies
- `requires_grad`, leaf gradients, and a per-result `GradNode`
- a scoped recording guard (`no_grad`) and `detach`
- reverse topological traversal with additive gradient accumulation
- backward rules for differentiable semantic primitives and public movement
  ops, not for composed public ops or physical layout variants
- one narrow internal `zero_pad` path supporting `shrink` backward
- a reusable central-difference gradient checker under CTest
- tests proving composed ops differentiate through the tape without dedicated
  rules
- tests proving graph lifetime and repeated-use behavior

**Supported scope:** float32 tensors, one explicit `core::context`, scalar-root
backward, one backward pass per recorded graph, and the differentiable ops named
in this module.

**Not required:** higher-order gradients, `retain_graph`, custom user-defined
backward functions, multithreaded backward, distributed autograd, a public
general scatter API, dedicated backward rules for composed ops, or in-place
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
- A raw view alias created inside a primitive's dispatch implementation is not
  a public movement op and receives no identity or node. In particular, binary
  kernel broadcast aliases are invisible to the tape; the binary node connects
  the original operands and reduces each gradient to its original shape once.
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

The fifth question forces an API decision because the current public movement
methods do not accept a context. The recommended first design is to make
differentiable tensor movement entry points accept `core::context&`, just like
arithmetic operations, while keeping raw `view` transformations available to
dispatch internals. An alternative tensor-associated recorder is acceptable
only if its owner, lifetime, and behavior across copied handles and contexts are
explicit. A thread-local or process-global tape is not acceptable.

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

Scalar constants used by composed ops, such as the reciprocal count inside
`mean`, are ordinary tensors from the tape's point of view. They must be created
with `requires_grad=false` so a composition never accumulates a gradient into
its own constant. Module 5A's primitive `neg` means `sub` needs no hidden
constant.

Use a visited set during topological traversal because a dynamic graph is a DAG,
not necessarily a tree. For `z = x*x + x`, the leaf `x` is reached through
multiple edges but its producing state must be visited once and its three local
contributions must be added.

**Tests:**

- `backward()` rejects a non-scalar root and a root that does not require grad
- recording is disabled inside `no_grad` and restored afterward
- a public movement operation observes the same context-scoped recording mode
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

**Self-test:** Run the checker first on a rank-0 function such as `x*x`, whose
derivative is `2*x`, and intentionally perturb one expected gradient to prove
the checker catches an error. Do not make this assignment depend on `sum`
backward, which is introduced in Assignment 6.5. Reuse the checker on vector
functions after reduction rules exist.

## Assignment 6.4: Primitive Elementwise Rules And Accumulation ⭐⭐⭐

**Task:** Add backward rules for all four primitive elementwise operations:

```text
add:  da = dout,             db = dout
mul:  da = dout * b,         db = dout * a
neg:  dx = neg(dout)
recip, with y = recip(x):    dx = neg(dout * y * y)
```

Build gradients from existing forward tensor ops while recording is disabled.
Every contribution is added to any existing contribution; never overwrite a
gradient merely because it is the first edge encountered during traversal.

For `recip`, save the forward result `y`; do not recompute the derivative as
`-dout/(x*x)`. The saved-output formula avoids overflow in `x*x` for large
finite inputs and needs no division during backward.

Then prove the composition claim: gradient-check `sub(a, b)` and `div(a, b)`
end to end **without writing any rule for them**. `sub` differentiates through
its recorded `neg` + `add` nodes; `div` through its recorded `recip` + `mul`
nodes. Keep denominators away from zero in the checker inputs, and use the
single documented gradient-checker tolerance policy rather than loosening it
only for division.

**Tests:** Gradient-check `add`, `mul`, `neg`, and `recip` on deterministic
rank-0 inputs so this assignment has no reduction dependency. Gradient-check
scalar `sub` and `div` through composition, and use `x*x + x` to prove
repeated-use accumulation. Broadcasted tensor cases follow after
`sum_to_shape` exists in Assignment 6.6.

## Assignment 6.5: Reduction Rules And Vector Gradient Checks ⭐⭐⭐

**Task:** Add gradients for full and axis `sum` before testing any vector-valued
operation with a scalar loss:

```text
sum(all):       expand dout to input shape
sum(axis):      reinsert the reduced axis when needed, then expand
```

The rule must honor Module 5's `keepdim` policy. Do not infer the reduced axis
later from ambiguous shapes; save the normalized axis, original input shape,
and whether the reduced dimension was retained.

Then prove the reduction composition: gradient-check full and axis `mean` end
to end without writing a `mean` rule. Reuse the checker on vector
`sum(x*x + x)` to prove repeated-use accumulation across a reduction.

**Tests:** Gradient-check full sum and full mean directly. For each axis of a
2-D sum or mean and both `keepdim` values, form a scalar test loss with an outer
full `sum`, such as `sum(mean(x, axis, keepdim))`, then run the checker and
assert the exact input-gradient shape. These rules establish the scalar-loss
path used by all remaining tensor-valued gradient checks.

## Assignment 6.6: Movement, Broadcasting, And Matmul Rules ⭐⭐⭐

**Task:** Implement these inverse mappings and the 2-D matmul rule:

| Forward op | Backward mapping |
| --- | --- |
| `reshape` | reshape to the input shape |
| `permute` | apply the inverse permutation |
| `expand` | sum over introduced and stride-zero axes |
| `shrink` | zero-pad `dout` to the input shape using the saved slice limits |
| `C = A @ B` | `dA = dout @ B^T`, `dB = A^T @ dout` |

Broadcasted binary nodes connect the original operands, not their internal raw
broadcast aliases. Their rules use the same `sum_to_shape` operation as public
`expand` backward. Given an upstream gradient and an original input shape,
`sum_to_shape` must reduce leading axes introduced by rank expansion and axes
where the original size was one, then restore the original rank. Add and mul
perform this reduction exactly once; no hidden `expand` node may repeat it.
When several axes must be reduced, either retain them with `keepdim=true` until
the final reshape or reduce non-kept axes in descending index order; otherwise
an early removal shifts later axis numbers and silently reduces the wrong
dimensions.

### The Missing Inverse Of `shrink`

Views can remove a rectangular region without moving data, but strides cannot
represent the inverse: backward must materialize zeros outside that region.
Add one narrow internal `zero_pad` submission path and Metal kernel:

```text
input:   dout, possibly strided
metadata: dout shape/strides/offset, original input shape,
          and saved [start, end) limits for each axis
output:  new contiguous tensor with the original input shape
kernel:  one thread per output element
         inside saved region -> read the corresponding dout element
         outside saved region -> write 0
```

This is not a public differentiable op and receives no GradNode because local
backward formulas execute under `no_grad` and higher-order gradients are out of
scope. Call it padding rather than a general scatter: rectangular shrink has
one source per in-region output, so it needs no overlap policy or atomics. A
future public padding or scatter API must earn its own semantics and tests.

**Required cases:**

- `(3,)` broadcast into `(4,3)`
- `(2,1,4)` broadcast into `(2,3,4)`
- transpose and general permute
- a non-zero-offset shrink
- contiguous reshape
- ordinary matmul and matmul with a transposed input view

Each case needs a numerical gradient check through the scalar `sum` path from
Assignment 6.5 and an exact gradient-shape check. Compare `zero_pad` directly
with a tiny CPU oracle for a multi-axis, non-zero-offset slice, and test invalid
saved limits before dispatch. Gradient-check broadcasted `add` and `mul` on
asymmetric shapes and verify both original gradient shapes exactly.

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
- [ ] 6.4 `add`/`mul`/`neg`/`recip` rules; scalar `sub`/`div` checked through composition.
- [ ] 6.5 Full/axis `sum` rules establish vector scalarization; full/axis `mean` is checked through composition.
- [ ] 6.6 Movement, broadcast reduction, and matmul rules; internal `zero_pad` supports `shrink`.
- [ ] 6.7 Graph lifetime and mutation boundary tests.
- [ ] 6.8 Magnetron copy/postpone comparison note.

## Exit Criteria

You are ready for Module 7 when:

1. Every differentiable semantic primitive (`add`, `mul`, `neg`, `recip`,
   `sum`, `matmul`, and the public movement ops) has a passing numerical
   gradient test and an exact gradient-shape assertion.
2. Every composed op (`sub`, `div`, full `mean`, and axis `mean`)
   gradient-checks through the tape with no dedicated rule, and that fact is
   covered by an explicit test.
3. Copies, branches, broadcasting, and repeated tensor use accumulate into the
   correct leaf identity.
4. `detach` and scoped `no_grad` have tested behavior.
5. Backward consumes and releases its graph, while leaf gradients persist until
   explicitly cleared.
6. Unsupported roots, repeated backward, and forbidden mutation fail clearly.
7. Internal binary broadcast aliases never create duplicate tape nodes or
   double-reduce gradients; a user-authored `expand` remains differentiable.
8. `shrink` backward uses the independently tested internal `zero_pad` path and
   restores the original shape and zero regions exactly.
9. You can trace one branched scalar loss by hand and match the implementation's
   topological order and accumulated values.
