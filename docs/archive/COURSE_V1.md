# Course V1 (Retired)

**Status:** Retired. Files removed from the worktree; recoverable from git.

comtam's first course was module-oriented: eighteen numbered modules across three phases, each with a contract, assignments, exit criteria, and a matching solution note carrying grading evidence. It was replaced on 2026-08-02 by the problem-driven chapter format in [`../course/INDEX.md`](../course/INDEX.md), because module boundaries kept splitting single hard mechanisms across several gates while bundling unrelated easy work into one.

## What It Covered

Phase I built the framework core: storage invariants, views and strides, eager dispatch, forward correctness against an oracle, broadcast/reduce/matmul, a mandatory primitive-surface consolidation gate, then autograd, nn modules and SGD, MNIST training, and core hardening. Phase II was a production track through public semantics, async execution, persistence, measured performance, a Python API, and release packaging. Phase III sequenced CNN, transformer, and GPT-2 capstones.

Modules 1 through 5A were completed and graded. Their outcome is the current implementation: the eager Metal runtime described in [`../ARCHITECTURE.md`](../ARCHITECTURE.md). Modules 6 onward were never started under V1; autograd is now Chapter 3 onward of the [autograd track](../course/autograd/INDEX.md).

## Recovering It

Every retired file still exists at `b9839283bef97c6b414b7b1072e6dd5d67676e7b`, the last commit before removal.

```sh
git show b9839283:docs/archive/course-v1/course/INDEX.md
git checkout b9839283 -- docs/archive/course-v1/
```

Paths under that revision:

```text
docs/archive/course-v1/course/     18 module contracts, INDEX, APPENDIX, PROJECTS
docs/archive/course-v1/solution/   assignment answers and grading for modules 1-5A
docs/archive/course-v1/AUTOGRAD_SPEC_DRAFT.md
```

The autograd spec draft is superseded by the chapter track and its [decision reference](../course/autograd/DECISIONS.md). Read it only for provenance; it does not describe the accepted design.
