## Sequential Capstone Track

The capstones prove that the framework mechanisms taught by the modules compose
into increasingly demanding real workloads. They are sequential rather than a
menu: pass each prerequisite module and capstone gate before beginning the next
stage.

Every capstone report separates:

```text
compiled    -> target built
smoke ran   -> one representative input completed
gate passed -> correctness, gradient, convergence/lifetime target met
```

Supporting work is attached to the capstone that earns it. Operator coverage is
not an independent feature sprint, the autograd visualizer supports debugging,
Python training repeats an already trusted native workload, and release
rehearsal proves artifact consumption.

## Capstone 1: MLP Training

**Prerequisite:** Module 8 for the native gate; Module 14 for the Python gate.

**Goal:** train the Module 8 two-layer MNIST MLP to greater than 90% validation
accuracy, first through the native API and later through the installed Python
API.

**Required evidence:** fixed seed and configuration, stable cross-entropy
oracle, multi-epoch accuracy, bounded live graph state, recorded batch/epoch
timing, and reproducible checkpoint resume. The Python run must use public
package symbols, preserve native parameter names and values, and exchange a
checkpoint with C++.

**Supporting milestone:** build a tiny autograd visualizer after Module 6. For
`sum(x*x + x)`, show node identity, saved inputs or metadata, topological order,
incoming gradient contributions, leaf accumulation, and graph-state release.
Enabling it must not change ownership or numerical results.

**Gate:** both required API surfaces reach the accuracy target from a clean
configuration, resume reproducibly, and exit without retained graph or native
state. The Python portion becomes mandatory only after Module 14.

## Capstone 2: CNN Training

**Prerequisites:** Module 16 and Capstone 1.

**Goal:** train one fixed small CNN on MNIST to greater than 98% validation
accuracy.

Module 16 earns only the operations this architecture needs, such as padding,
`conv2d`, pooling, activation functions, and any required batched operation. Add
them one at a time with forward and gradient oracles; do not begin with a
general convolution subsystem.

**Gate:** deterministic multi-epoch convergence, complete supported-semantics
rows for every new operation, bounded lifetime behavior, native/Python agreement,
and a measured performance comparison identifying the dominant CNN cost.

## Capstone 3: GPT-2

**Prerequisites:** Module 17 and Capstone 2 for tiny training; Module 18 and the
tiny-training gate for real-checkpoint inference.

**Goal:** first prove transformer training semantics on a tiny GPT-2-shaped
model, then load a real pretrained GPT-2 SafeTensors checkpoint and reproduce
reference inference results.

### Tiny GPT-2 Training Gate

Train a deliberately small transformer to overfit a fixed tiny corpus. Compare
forward intermediates, loss, selected parameter gradients, and update results
against an independent reference implementation. This gate proves that the
transformer operations and autograd rules compose; it does not claim practical
GPT-2-scale training.

### Real GPT-2 Inference Gate

Use Python SafeTensors tooling to validate, rename, transpose, and convert
external weights as necessary, then load them through comtam's public Python
API. Initially keep SafeTensors parsing, tokenization, model download, and
checkpoint conversion outside the C++ tensor core.

On fixed token IDs, compare selected intermediate activations, final logits,
and deterministic generated token IDs with a pinned trusted reference. Record
checkpoint identity, conversion rules, tolerances, memory use, and end-to-end
latency.

Native SafeTensors support or full-size GPT-2 training requires a separate
measured product need. Neither is part of this capstone gate.

## Release Candidate Rehearsal

**Prerequisite:** Module 15. Repeat after each capstone that broadens the claimed
supported workload.

Prove another clean machine or environment can consume native and Python
artifacts alone. Install the release candidate, run documentation examples,
load the currently supported model, execute its acceptance scenario, run
compatibility fixtures, and archive benchmark/version reports with checksums.

This rehearsal is a supporting product gate rather than a fourth model
capstone. A successful Module 15 rehearsal covers the MLP-era product; CNN and
GPT-2 support are not release claims until their own rehearsals pass.

## Scope Boundaries

The sequential track does not authorize broad API accumulation. Every new
operation needs a real capstone call site, an independent forward oracle, a
numerical gradient test when differentiable, scalar/empty/layout coverage, and
Python coverage after Module 14.

Multiple backends, distributed execution, lazy graphs, graph compilers, broad
PyTorch/NumPy compatibility, tokenization, and model downloading remain outside
the framework core. Add a second dtype only when its storage, transfer,
validation, operator, persistence, and Python semantics are specified and
tested together.
