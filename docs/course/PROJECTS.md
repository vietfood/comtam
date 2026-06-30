## Final Projects

Final projects prove the eager stack works end to end. Choose one only after
Modules 1 through 7 are stable. Module 8 is strongly recommended first, because
it shapes the training loop every project depends on.

Each project ends with a gate, not a demo. Separate "compiled", "smoke ran", and
"gate passed".

### Project A: MNIST MLP ⭐⭐

**Goal:** Train a 2-layer MLP on MNIST to >90% accuracy.

**Requirements:**

- `nn::Linear` (Module 7)
- `relu` and softmax cross-entropy (Modules 7 and 8)
- `SGD` (Module 7)
- MNIST loading (Module 8)

**Steps:**

1. Assemble `Linear -> relu -> Linear`.
2. Implement the epoch/batch training loop.
3. Track loss and validation accuracy.
4. Verify convergence across real epochs.

**Why this project:** The MNIST MLP is the smallest proof that forward ops,
autograd, the optimizer, and data movement are coherent enough to train. It is
mostly Module 8 made rigorous.

---

### Project B: MNIST CNN ⭐⭐⭐

**Goal:** Train a small CNN on MNIST to >98% accuracy.

**Requirements (builds on Project A):**

- `View` padding and `shrink` working solidly (Module 2)
- `nn::Conv2d` via im2col
- `nn::MaxPool2d`
- batched matmul (extend Module 5)

This is significantly harder because Conv2d requires:

- padding for same-size convolutions
- sliding-window extraction (im2col)
- batched matmul for the actual convolution
- shrink for output sizing

**Why this project:** CNNs force the view system to grow up. If padding, shrink,
reshape, and batched matmul are shaky, this project exposes it fast. Expect to
revisit Module 2 and add the `pad` op you previously deferred.

---

### Project C: Tiny Autograd Visualizer ⭐⭐

**Goal:** Build a tool that prints or renders the dynamic autograd graph for a
forward+backward pass.

Show:

- the grad nodes created during forward
- the topological order used by backward
- which tensors accumulate gradients from more than one path
- saved tensors per node and when they are released

**Why this project:** comtam's autograd graph is built eagerly and is invisible by
default. Making it visible is a strong test of whether you understand Module 6.
If you can render the graph and explain every edge, you understand the backward
pass.

---

### Project D: Operator Coverage Sprint ⭐⭐

**Goal:** Widen op coverage with full forward and gradient tests for each.

Candidates: `exp`, `log`, `sqrt`, `sigmoid`, `tanh`, `max`/`min` reductions,
`where`.

Rules:

- Each new op needs a CPU-oracle forward test and a numerical gradient test
  before it counts.
- Add ops only with the dispatch and autograd patterns already established. If an
  op does not fit, record that design pressure in the project notes instead of
  special-casing it.

**Why this project:** Breadth with discipline. This project rewards the test
infrastructure from Modules 4 and 6 and reveals where the op/dispatch/autograd
patterns are still too rigid.

---

Good luck. The best way to learn is to implement, get stuck, then discover why
the proper solution works better. Keep patches small and keep the gates honest.
