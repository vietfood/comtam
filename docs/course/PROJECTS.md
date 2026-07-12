## Workload Projects

Projects earn framework breadth by applying the smallest existing runtime to a
real pressure. They do not override module order: finish the prerequisite gate,
then choose one project whose missing capability you actually want to support.

Every project report separates:

```text
compiled    -> target built
smoke ran   -> one representative input completed
gate passed -> correctness, gradient, convergence/lifetime target met
```

## Core Projects

### Project A: MNIST MLP ⭐⭐

**Prerequisite:** Module 8.

**Goal:** reproduce the Module 8 two-layer MLP at greater than 90% validation
accuracy from a clean configuration.

**Required evidence:** fixed seed/configuration, stable cross-entropy oracle,
multi-epoch accuracy, bounded live graph state, and recorded batch/epoch timing.

This is the recommended first project because it proves parameters, forward,
autograd, optimizer, data movement, and sustained execution together.

### Project B: Tiny Autograd Visualizer ⭐⭐

**Prerequisite:** Module 6.

**Goal:** render or print the dynamic graph for a branched forward/backward pass.

Show node identity, saved inputs/metadata, topological order, each incoming
gradient contribution, leaf accumulation, and when graph state is released.

**Gate:** the visualization of `sum(x*x + x)` matches a hand trace, and enabling
the visualizer does not change ownership or numerical results.

### Project C: Operator Coverage Sprint ⭐⭐

**Prerequisites:** Modules 6 and 10.

**Goal:** add a workload-driven subset of `exp`, `log`, `sqrt`, `sigmoid`, `tanh`,
max/min reductions, comparisons, or `where`.

Each op needs a supported-semantics matrix row, independent forward oracle,
numerical gradient test when differentiable, scalar/empty/layout cases, and
Python coverage if Module 14 has passed. If an op does not fit the existing
dispatch/autograd pattern, document the design pressure before special-casing it.

## Model-Expansion Projects

### Project D: MNIST CNN ⭐⭐⭐

**Prerequisites:** Modules 9-10 and Project A.

**Goal:** train a small CNN on MNIST to greater than 98% validation accuracy.

This project may earn padding, window extraction/im2col, batched matmul,
`conv2d`, and pooling. Add them one at a time with forward/gradient oracles; do
not start by building a general convolution subsystem.

**Gate:** deterministic multi-epoch convergence, complete new-op conformance
rows, and a performance comparison identifying whether im2col materialization or
matmul dominates.

### Project E: Tiny Transformer Inference ⭐⭐⭐

**Prerequisites:** Modules 10-13.

**Goal:** run one small, fixed transformer-like inference workload from a saved
checkpoint.

Expected pressure includes batched matmul, embeddings/gather, softmax, layer
normalization, integer token indices, causal masking, and persistence. This
project earns an integer dtype only when its storage, transfer, validation,
operator, serialization, and Python semantics are all implemented.

**Gate:** compare intermediate activations and final logits to an independent
reference on fixed tokens, then record memory and latency for the complete
sequence.

Do not add tokenization or model downloading to the tensor core. Conversion and
fixtures belong in tools/examples.

## Product Projects

### Project F: Python Training Package ⭐⭐⭐

**Prerequisite:** Module 14.

**Goal:** reproduce Project A through the installed Python API without calling
internal extension symbols.

**Gate:** a clean environment installs the built wheel, trains or loads the MLP,
passes the accuracy target, saves a checkpoint that C++ can read, and exits
without retained native state. NumPy is the forward oracle; native and Python
parameter names/values must agree.

### Project G: Release Candidate Rehearsal ⭐⭐⭐

**Prerequisite:** Module 15.

**Goal:** prove another machine/environment can consume the narrow supported
runtime from artifacts alone.

**Gate:** native package and Python wheel install from the release candidate,
documentation examples run, the supported model loads and infers, compatibility
fixtures pass, and benchmark/version reports are archived with checksums.

## Choosing A Project

Choose the smallest project that forces a capability you genuinely want. Project
A validates the core, Project B deepens autograd understanding, Project C widens
operators, Projects D/E widen model scope, and Projects F/G validate the Python
and release surfaces. Completing all of them is not a course requirement.
