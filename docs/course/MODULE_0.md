# Module 0: Scope And Skeleton

This module prevents fake progress. Before kernels, autograd, or `nn.Linear`,
write down what `comtam` is allowed to be.

## What To Build

Create a minimal CMake project with a colocated source/header layout:

```text
comtam/
  core/
    core.h
    core.cpp
tests/
vendor/
  metal/
```

Add placeholder types:

- `Context`
- `Device`
- `Storage`
- `Tensor`
- `OpCode`

Do not implement a tensor API yet. The point is to set the boundaries before
they get blurry.

## Design Notes

`refs/magnetron` is the architecture reference, but it is larger than `comtam`
should be. Borrow the arc, not the size:

```text
context -> tensor/storage -> device command -> autograd -> nn/optim
```

Reject these features for now:

- Python bindings
- multiple backends
- dynamic backend loading
- many dtypes
- serialization
- image/audio IO
- allocator caches

## Assignment 0.1: Write The Project Contract

Create a short document or comment block answering:

1. What is the smallest training loop this repo must support?
2. Which C++ features are allowed?
3. Which features are explicitly out of scope?
4. What does "done" mean for the first end-to-end example?

## Assignment 0.2: CMake Smoke Test

Set up a test executable that:

1. Creates a `Context`.
2. Initializes the default Metal device.
3. Creates a command queue.
4. Tears everything down without leaking.

## Completion Gate

You may start Module 1 only when:

- `cmake --build` succeeds.
- A smoke test constructs and destroys the Metal context.
- The repo has no tensor API beyond placeholders.
- The project contract says what is deliberately not being built.
