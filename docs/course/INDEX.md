# comtam Course

This course is a self-guided path for building `comtam`, first as a correct tiny
eager-mode deep learning framework in C++20 for Apple devices, then as a narrow
production-quality runtime with a Python API.

The point is not to copy PyTorch or Magnetron. The point is to rebuild enough of
their ideas that you understand why they exist, feel the cost of each abstraction
before adopting it, and eventually support one explicit workload with evidence.

## Course Thesis

comtam stays Metal-only, float32-first, single-device, and eager. A tensor op
runs now; the runtime does not build a lazy forward graph, schedule fusion, or
hide device ownership behind a global manager.

The architectural path is:

```text
Phase I: Framework Core

Metal context
  -> storage and tensor metadata
  -> views and strides
  -> eager operator dispatch
  -> forward correctness
  -> broadcasting, reductions, matmul
  -> reverse-mode autograd
  -> nn modules and SGD
  -> sustained training
  -> core semantic hardening

Phase II: Production Track

supported public semantics
  -> async execution and memory lifetime
  -> persistence and reproducibility
  -> measurement-driven performance
  -> Python API
  -> packaging, CI, and release evidence
```

Phase I teaches how a tensor framework works. Phase II teaches why a working
framework is still not a supported product.

## Scope Rules

The following are deferred during Modules 1-9, not rejected forever:

- Python bindings, earned in Module 14 after the C++ API is stable.
- Serialization, earned in Module 12 after parameter names and mutation rules
  are stable.
- Async submission and allocator reuse, considered in Module 11 only after
  lifetime tests and measurements exist.
- Detailed performance work, earned in Module 13 from a real training baseline.
- A second dtype, earned only by a concrete workload with complete storage,
  transfer, validation, op, and persistence semantics.

These remain outside the complete course unless a new product goal changes the
scope:

- multiple backends or dynamic backend loading
- distributed or multi-device execution
- lazy graphs, graph compilers, scheduling, and fusion
- broad PyTorch/NumPy API compatibility
- image/audio IO inside the framework core

The reasoning and legrad counterexamples live in
[`../note/AVOID.md`](../note/AVOID.md).

## How To Use Each Module

Every module now has a **Module Contract**. Read that before assignments; it
defines prerequisites, required deliverables, supported scope, non-goals, tests,
and completion evidence.

Use this workflow:

1. Read the module contract and copy its checklist into the matching
   `docs/solution/MODULE_N.md` file.
2. Attempt each assignment before reading the reference framework.
3. Write down the API/semantic decisions the assignment explicitly requests.
4. Implement the smallest version satisfying the listed behavior.
5. Add the required tests and independent oracle before calling the assignment
   complete.
6. Record exact commands and results, separating build, smoke, and gate evidence.
7. Compare with Magnetron only after your implementation works or you have a
   concrete design question.

Instruction vocabulary is deliberate:

- **Task** says what behavior to implement.
- **You must decide** identifies a semantic choice that belongs in the solution
  note before coding.
- **Minimum implementation** bounds the accepted first version.
- **Tests** are mandatory gate evidence unless labeled recommended/historical.
- **Not required** is a scope boundary, not optional extra credit.
- **Exit Criteria** are the only authority for starting the next module.

Difficulty ratings:

- ⭐ Straightforward
- ⭐⭐ Requires careful thinking
- ⭐⭐⭐ Challenging, multiple interacting concerns

## Phase I: Framework Core

Follow this order even if a later module looks more interesting.

1. [`MODULE_1.md`](MODULE_1.md) - storage invariants and tensor metadata ownership
2. [`MODULE_2.md`](MODULE_2.md) - master views, strides, offsets, and indexing
3. [`MODULE_3.md`](MODULE_3.md) - eager operator dispatch to Metal
4. [`MODULE_4.md`](MODULE_4.md) - forward correctness against an external oracle
5. [`MODULE_5.md`](MODULE_5.md) - broadcasting, reductions, and matmul
6. [`MODULE_6.md`](MODULE_6.md) - dynamic reverse-mode autograd and graph lifetime
7. [`MODULE_7.md`](MODULE_7.md) - stable parameters, modules, and SGD
8. [`MODULE_8.md`](MODULE_8.md) - sustained end-to-end MNIST training
9. [`MODULE_9.md`](MODULE_9.md) - core semantic, error, and lifetime hardening

Why this order:

- Storage ownership and byte sizing poison every later lifetime question when
  they remain vague.
- View semantics precede dispatch because wrong strides produce convincing but
  wrong kernels.
- One eager op runs end to end before dispatch is generalized.
- Forward oracles precede gradient rules so autograd starts from trusted math.
- Broadcast, reduction, and matmul supply the shape-changing primitives needed
  by backward and layers.
- Autograd defines identity, recording, accumulation, and graph lifetime before
  an optimizer mutates parameters.
- Modules and SGD prove those semantics compose into stable trainable state.
- Sustained training exposes numerical and lifetime failures invisible to one
  step.
- Module 9 freezes the supported core before external APIs depend on it.

Phase I completion means **correct educational eager runtime**, not production
readiness.

## Phase II: Production Track

10. [`MODULE_10.md`](MODULE_10.md) - supported public tensor semantics
11. [`MODULE_11.md`](MODULE_11.md) - execution, synchronization, and memory lifetime
12. [`MODULE_12.md`](MODULE_12.md) - persistence and reproducible resume
13. [`MODULE_13.md`](MODULE_13.md) - measurement-driven performance
14. [`MODULE_14.md`](MODULE_14.md) - native-backed Python API
15. [`MODULE_15.md`](MODULE_15.md) - packaging, CI, compatibility, and release

Why Python is Module 14 rather than an early convenience layer: bindings expose
and freeze ownership, exceptions, mutation, synchronization, persistence, and
operator semantics. Building them after those contracts exist keeps Python thin
and prevents a second framework implementation from growing in wrapper code.

Phase II completion supports a narrow claim: production-quality for the tested
Apple Silicon, Metal, float32, eager workload. It does not imply general-purpose
PyTorch or Magnetron feature parity.

## Module Map

| Module | Topic | Gate question |
| --- | --- | --- |
| [`MODULE_1.md`](MODULE_1.md) | Storage invariants | Who owns GPU bytes and can tensor headers share them safely? |
| [`MODULE_2.md`](MODULE_2.md) | Views and strides | Does logical indexing map to the intended storage element? |
| [`MODULE_3.md`](MODULE_3.md) | Eager dispatch | Does one validated op launch the correct Metal kernel? |
| [`MODULE_4.md`](MODULE_4.md) | Forward correctness | Do values and shapes match an independent oracle? |
| [`MODULE_5.md`](MODULE_5.md) | Broadcast/reduce/matmul | Do shape-changing ops work on required layouts and edges? |
| [`MODULE_6.md`](MODULE_6.md) | Autograd | Do graph identity, lifetime, and local gradients compose? |
| [`MODULE_7.md`](MODULE_7.md) | nn and SGD | Can stable leaf parameters learn without recording updates? |
| [`MODULE_8.md`](MODULE_8.md) | Training | Does a real sustained workload converge without leaked state? |
| [`MODULE_9.md`](MODULE_9.md) | Core hardening | Is every supported and unsupported core behavior explicit? |
| [`MODULE_10.md`](MODULE_10.md) | Public semantics | Can external C++ code rely on alias, mutation, shape, and API rules? |
| [`MODULE_11.md`](MODULE_11.md) | Execution and memory | Should work stop blocking, and can the selected model preserve resources/errors? |
| [`MODULE_12.md`](MODULE_12.md) | Persistence | Can state round-trip, reject corruption, and resume reproducibly? |
| [`MODULE_13.md`](MODULE_13.md) | Performance | Is each kept optimization supported by repeatable measurement? |
| [`MODULE_14.md`](MODULE_14.md) | Python API | Does Python preserve the native runtime's identity and semantics? |
| [`MODULE_15.md`](MODULE_15.md) | Release | Can clean external consumers install and run supported artifacts? |

## Progress

| Module | Status | Evidence | Notes |
| --- | --- | --- | --- |
| [`MODULE_1.md`](MODULE_1.md) | Passed | `cmake -S . -B build -DCOMTAM_BUILD_TESTS=ON`; build; CTest | Verified 2026-06-30; grading in [`../solution/MODULE_1.md`](../solution/MODULE_1.md). |
| [`MODULE_2.md`](MODULE_2.md) | Passed | build; CTest | Verified 2026-06-30; deferred 2.9 is non-blocking; grading in [`../solution/MODULE_2.md`](../solution/MODULE_2.md). |
| [`MODULE_3.md`](MODULE_3.md) | Passed | build; CTest | Verified 2026-07-01; grading in [`../solution/MODULE_3.md`](../solution/MODULE_3.md). |
| [`MODULE_4.md`](MODULE_4.md) | Passed | build; CTest | Verified 2026-07-01; grading in [`../solution/MODULE_4.md`](../solution/MODULE_4.md). |
| [`MODULE_5.md`](MODULE_5.md) | In progress | broadcast/matmul tests; reduction WIP | Gate remains open; grading in [`../solution/MODULE_5.md`](../solution/MODULE_5.md). |
| Modules 6-15 | Not started | - | Start only after the preceding module gate passes. |

Keep this table compact. Assignment-level answers, commands, grading, missing
tests, and pass/fail reasoning belong in the matching `docs/solution/MODULE_N.md`.

## Supporting Material

- [`APPENDIX.md`](APPENDIX.md) - code map, reference-reading order, and debugging
- [`PROJECTS.md`](PROJECTS.md) - workload projects that earn new framework scope
- [`../ARCHITECTURE.md`](../ARCHITECTURE.md) - current implementation architecture
- [`../note/AVOID.md`](../note/AVOID.md) - premature abstractions to avoid
- [`../note/METAL_USAGE.md`](../note/METAL_USAGE.md) - Metal ownership and usage
