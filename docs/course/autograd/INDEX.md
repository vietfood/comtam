# Autograd Track

This track teaches how comtam grows from a correct eager Metal tensor runtime into a correct eager tensor runtime with dynamic reverse-mode autograd. It begins with runtime lifetime because every later graph node and backward kernel depends on knowing where execution resources come from and who keeps them alive.

## Track Outcome

By the end of the track, you should be able to explain and implement:

- tensor-bound runtime identity with a process-wide default convenience;
- stable logical tensor identity across C++ handle copies;
- public movement identities that share storage without sharing autograd identity;
- thread-local, runtime-specific recording suppression;
- output-owned polymorphic backward nodes without shared-pointer cycles;
- reverse topological traversal and additive gradient accumulation;
- detached saved values, storage-version checks, and graph consumption;
- numerical gradient checks for every supported differentiable primitive.

## Chapter Sequence

| Chapter | Problem | Status |
| --- | --- | --- |
| [1. Runtime Ownership Before Autograd](01_RUNTIME_OWNERSHIP.md) | Where do operations and future backward rules obtain Metal resources, and who keeps them alive? | Passed |
| [2. Tensor Identity And `tensor_impl`](02_TENSOR_IDENTITY.md) | How can `tensor b = a` preserve one gradient identity while movement creates a new identity that shares storage? | Passed |
| [3. Recording And `no_grad`](03_RECORDING_AND_NO_GRAD.md) | When does an eager operation record, and how does suppression stay nested, thread-local, and runtime-specific? | Ready to implement |
| 4. Graph Ownership And `grad_fn` | How do heterogeneous backward rules retain parents and saved values without retaining their own outputs? | Planned |
| 5. The Backward Engine | How does a DAG traverse once per identity, accumulate every edge, commit leaf gradients, and consume successfully? | Planned |
| 6. Saved Values And Mutation | Which forward values must survive, and how does backward reject values changed after recording? | Planned |
| 7. Primitive Elementwise Gradients | How do local rules compose through eager primitives and numerical gradient checks? | Planned |
| 8. Reduction And Broadcasting Gradients | How are upstream gradients expanded or reduced back to exact original shapes? | Planned |
| 9. Movement And Matmul Gradients | How do inverse view mappings, zero padding, and transposed matmul restore input gradients? | Planned |

## How To Use This Track

Read chapters sequentially because each establishes an ownership or execution invariant the next one assumes. Implement only the checkpoint currently being studied; a broad autograd rewrite would make runtime, identity, graph, and derivative failures indistinguishable.

Chapters 1 and 2 are retrospective - the mechanisms they describe are in the source, and they cite it by file and line rather than reproducing it. Read them for the reasoning, and read the code beside them. Chapter 3 onward is written before the implementation exists and carries an exercise.

Chapters contain an implementation exercise rather than an implementation patch. Code sketches are deliberately partial - just enough to make a type or an ownership relation concrete. A chapter has done its job only when you can explain why the implementation is shaped that way.

The compact accepted choices live in [`DECISIONS.md`](DECISIONS.md). Read that file as a reference after the relevant tutorial chapter explains a decision; it is not a substitute for the chapters.

## Prerequisite State

The active source already has tested eager storage, views, semantic primitive dispatch, broadcasting, reductions, and matmul. [`../../archive/COURSE_V1.md`](../../archive/COURSE_V1.md) records the retired module path that produced them.
