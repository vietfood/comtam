## Module 6: Reverse-Mode Autograd

Build and verify the backward pass.

## Why This Module Comes After Forward Correctness

Autograd is a second graph system layered on top of the forward tensor system.

In comtam's eager design there is no forward graph sitting around to differentiate
later. Instead, each differentiable op records, at the moment it runs, enough
information to compute its local gradient. Backward then walks that recorded
history in reverse.

If forward behavior is unstable, autograd amplifies the instability. That is why
Modules 4 and 5 came first. The goal here is not a full PyTorch autograd clone.
It is to understand reverse-mode autodiff in an eager tensor framework:

```text
op runs forward
  -> record a grad node (op, inputs, saved values)
  -> build output Tensor that points at that node
backward(loss)
  -> topological order over the dynamic node graph
  -> seed loss gradient = 1
  -> walk in reverse, apply local rules, accumulate into input grads
```

## Design From The Architecture

[`../ARCHITECTURE.md`](../ARCHITECTURE.md) already sketches the eager autograd
shape. comtam should store gradient nodes on tensors, not introduce a separate
tape abstraction yet:

```text
Tensor adds:
  requires_grad
  grad           (a Tensor, the accumulated gradient)
  grad_node      (how this tensor was produced)

GradNode:
  op
  inputs (the tensors this one was computed from)
  saved values needed by the backward rule
```

Backward procedure:

1. Require a scalar root (the loss).
2. Topologically sort the dynamic graph reachable from the root.
3. Seed the root gradient with ones.
4. Traverse in reverse topological order.
5. Call each node's local backward rule.
6. Accumulate gradients into inputs (`+=`, never overwrite).

Do not introduce a separate tape abstraction until storing nodes on tensors is
clearly insufficient.

## The Gradient Test Is The Module

A backward rule that looks right can still be wrong once composed. The deliverable
of this module is a gradient-checking harness, not just rules.

Two oracles:

- **analytic CPU oracle:** a hand-derived gradient for small fixed inputs.
- **numerical gradient (finite differences):** perturb one input by `eps`,
  measure the change in the scalar loss, and compare to the analytic gradient.

```text
grad_numeric[i] ~= (loss(x + eps*e_i) - loss(x - eps*e_i)) / (2*eps)
```

Central differences with `eps` around `1e-3` to `1e-4` for float32, compared with
a loose tolerance, catch most rule and composition bugs.

## Assignment 6.1: Design The Grad Node ⭐⭐

**Task:** Decide the concrete representation before writing rules.

Questions:

1. Does `grad_node` live on the `Tensor` (as in `tensor.h`) or in a side
   structure? Start on the tensor, per the architecture.
2. What does each op need to save for its backward? (`mul` needs both inputs;
   `add` needs nothing but the output shapes; matmul needs both inputs.)
3. How do you avoid keeping the entire graph alive forever? When is it safe to
   drop saved tensors?

**Deliverable:** A one-page design note: node fields, ownership (who keeps saved
tensors alive), and how backward finds the topological order.

## Assignment 6.2: Implement Backward For Elementwise Ops ⭐⭐

**Task:** Add backward rules for `add`, `sub`, `mul`, `div`.

Local rules:

```text
add:  da = dout,        db = dout
sub:  da = dout,        db = -dout
mul:  da = dout * b,    db = dout * a
div:  da = dout / b,    db = -dout * a / (b*b)
```

Rules:

- Accumulate (`+=`) into input grads, because an input may be used more than once.
- Each backward is itself built from forward ops you already have. Backward is
  just more tensor math.

**Test:** Numerical gradient check on each op for small asymmetric inputs, plus a
case where the same tensor is used twice (for example `x * x`) to verify
accumulation.

## Assignment 6.3: Implement Backward For Movement Ops ⭐⭐

**Task:** Add gradients for the Module 2 view ops, using the pairings from
Assignment 2.8:

| Forward op | Backward op |
| --- | --- |
| `reshape` | `reshape` back to input shape |
| `permute` | inverse `permute` |
| `expand` | `sum` over expanded (stride-0) axes |
| `slice` / `shrink` | scatter gradient into a zero tensor at the slice |

**Why this matters:** Movement gradients teach inverse views. The backward rule
for a shape op is usually another shape op, not a numeric kernel. The one
exception here is `expand`, whose backward is a reduction (Assignment 6.4).

**Test:** Numerical gradient check for transpose, reshape, and slice.

## Assignment 6.4: Broadcasting Gradient Accumulation ⭐⭐⭐

**Task:** Make broadcasted ops produce correctly shaped gradients.

When a tensor is broadcast in the forward pass, its gradient must be summed back
to the original shape.

```text
a: (3,)      b: (4, 3)
c = a + b    -> c: (4, 3)
loss = sum(c)
a.grad must be (3,), summed over the broadcast axis
```

Questions:

1. Which axes were introduced by broadcasting (rank difference)?
2. Which axes had size 1 and expanded to a larger size?
3. Does your `expand` backward reduce over both cases?

**Why this assignment:** Broadcasting is where many toy autograd engines quietly
lie. If this is wrong, training can run while producing bad gradients and you
will blame the optimizer.

**Test:** The `(3,) + (4, 3)` case above, checked numerically, plus a size-1
middle-axis case like `(2, 1, 4) + (2, 3, 4)`.

## Assignment 6.5: Backward For Reductions And Matmul ⭐⭐⭐

**Task:** Add gradients for `sum`, `mean`, and 2-D `matmul`.

Rules:

```text
sum (all):   da = dout broadcast back to input shape
sum (axis):  da = dout expanded along the reduced axis
mean:        like sum, scaled by 1/count
matmul C=A@B: dA = dout @ B^T,   dB = A^T @ dout
```

Notice that matmul backward is two more matmuls with transposed views. This is
why Module 2's `transpose` and Module 5's strided-input matmul had to come first.

**Test:** Numerical gradient checks for `sum`, `mean(axis)`, and a small matmul.

## Assignment 6.6: Read Magnetron Autograd ⭐⭐⭐

After your own attempt, read Magnetron's dynamic autograd.

Questions:

1. How does Magnetron record the backward graph during eager execution?
2. How does it decide topological order?
3. How does it accumulate gradients when a tensor is reused?
4. Which part of its design would be premature for comtam now?

**Deliverable:** A short note on what comtam's autograd should copy and what it
should postpone.

## Module 6 Checklist

- [ ] 6.1 Design and document the grad node representation.
- [ ] 6.2 Backward for add/sub/mul/div with accumulation.
- [ ] 6.3 Backward for movement ops.
- [ ] 6.4 Broadcasting gradient accumulation.
- [ ] 6.5 Backward for sum/mean/matmul.
- [ ] 6.6 Read Magnetron autograd; write a copy/postpone note.

## Exit Criteria

You are ready for Module 7 when:

1. A numerical gradient-check harness exists and runs under CTest.
2. Every differentiable op has a passing gradient test.
3. Broadcasting gradients reduce back to the correct input shape, verified.
4. Gradients accumulate correctly when a tensor is used more than once.
5. You can trace one scalar loss backward by hand, node by node.
