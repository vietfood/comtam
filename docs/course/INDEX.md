# comtam Learning Course

This is the active course for learning how comtam works by designing and implementing it. The course is organized around difficult mechanisms rather than broad modules. A mechanism may need several chapters, and each chapter develops the problem deeply enough that you can implement it without hidden architectural leaps.

## Course Method

Every chapter follows the same learning arc:

```text
concrete problem in current comtam
-> mental model
-> plausible designs
-> consequences and rejected alternatives
-> chosen design
-> ownership and lifetime trace
-> concrete C++ types
-> incremental implementation checkpoints
-> tests proving the mechanism
-> common mistakes
-> implementation exercise
-> chapter completion gate
```

The goal is not to make each assignment easy. The goal is to make every difficult step visible, motivated, and testable. A chapter may deliberately teach advanced C++ ownership or runtime techniques when they solve a real comtam problem.

## Active Tracks

| Track | Purpose | Status |
| --- | --- | --- |
| [`autograd/`](autograd/INDEX.md) | Runtime identity, tensor identity, graph ownership, backward execution, and gradient rules | Chapters 1-2 ready |

New subjects such as neural-network modules, optimizers, sustained training, persistence, performance, and Python bindings will become their own multi-chapter tracks when the implementation reaches them.

## How To Study A Track

1. Read one chapter completely before editing code.
2. Redraw the ownership or execution diagram in your own words.
3. Implement only the chapter's current checkpoint.
4. Run the exact checks named by the chapter.
5. Record anything that behaved differently from the chapter's prediction.
6. Do not continue until you can explain the chapter completion gate without reading it verbatim.

The course is living guidance. Verified code and tests may justify a better design, but the chapter and its decision reference must be updated deliberately so later work does not inherit an unexplained divergence.

## Historical Course

The original module-oriented course and grading history are frozen under [`../archive/course-v1/`](../archive/course-v1/README.md). Use that archive for provenance and earlier exercises, not as a blocker against the active problem-driven course.
