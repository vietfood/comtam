# comtam Self-Study Course

This course is for building `comtam`: a tiny eager-mode deep learning framework
in C++20 for Apple devices.

`refs/banhxeo` is only a reference. It is a different framework and a different
course: compiler-first, lazy, Triton-based, and Python-facing. `comtam` should
stay runtime-first: eager tensors, explicit Metal execution, dynamic autograd,
and a small enough codebase that the whole path from `Tensor` to kernel launch
can fit in your head.

## Course Thesis

The shortest honest path is:

```text
Metal context
  -> storage and tensor metadata
  -> views and strides
  -> eager operator dispatch
  -> broadcasting, reductions, matmul
  -> reverse-mode autograd
  -> nn modules and SGD
  -> small training examples
  -> hardening only after correctness
```

The goal is not to compete with PyTorch, MLX, or Magnetron. The goal is to write
one tiny framework where every abstraction earns its place.

## Constraints

- C++20.
- Metal via `metal-cpp` is the only external dependency.
- Start with `float32` only.
- Start with one Apple GPU device.
- Start with shared Metal buffers.
- Prefer direct, boring code over reusable-looking machinery.
- No ATen wrappers, no Python binding layer, no lazy graph, no multi-backend
  registry, no custom allocator until profiling proves it matters.

## Reference Roles

- `refs/magnetron`: architectural reference for tensor metadata, view ownership,
  operator dispatch, and dynamic autograd.
- `refs/legrad`: anti-pattern reference. It grew framework-shaped abstractions
  before a small end-to-end training loop was stable.
- `refs/banhxeo`: contrast reference. Useful for views and tensor semantics, but
  its lazy compiler path is not the architecture of `comtam`.

## How To Use This Course

1. Do the modules in order.
2. Do not read ahead as implementation permission.
3. Each module has a completion gate.
4. Do not start module N+1 until module N's gate is met.
5. When stuck, write the smallest repro, not a broad redesign.

## Recommended Order

1. [`MODULE_0.md`](MODULE_0.md) - scope and repository skeleton
2. [`MODULE_1.md`](MODULE_1.md) - Metal ownership and storage
3. [`MODULE_2.md`](MODULE_2.md) - tensor metadata, shapes, strides, and views
4. [`MODULE_3.md`](MODULE_3.md) - eager operator dispatch
5. [`MODULE_4.md`](MODULE_4.md) - broadcasting and reductions
6. [`MODULE_5.md`](MODULE_5.md) - matmul and layout
7. [`MODULE_6.md`](MODULE_6.md) - reverse-mode autograd
8. [`MODULE_7.md`](MODULE_7.md) - modules, parameters, losses, and optimizers
9. [`MODULE_8.md`](MODULE_8.md) - hardening without bloat
10. [`PROJECTS.md`](PROJECTS.md) - end-to-end proofs

## Module Map

| Module | Topic | Completion Gate |
| --- | --- | --- |
| [`MODULE_0.md`](MODULE_0.md) | Scope and skeleton | CMake project creates a context smoke test |
| [`MODULE_1.md`](MODULE_1.md) | Metal ownership | Host-device roundtrip works leak-free |
| [`MODULE_2.md`](MODULE_2.md) | Views and strides | Logical indices map to expected storage offsets |
| [`MODULE_3.md`](MODULE_3.md) | Op dispatch | Elementwise ops share one dispatch path |
| [`MODULE_4.md`](MODULE_4.md) | Broadcast/reduce | Broadcast, sum, and mean match oracles |
| [`MODULE_5.md`](MODULE_5.md) | Matmul | `x @ w + b` matches an oracle |
| [`MODULE_6.md`](MODULE_6.md) | Autograd | Gradient checks pass for core ops |
| [`MODULE_7.md`](MODULE_7.md) | NN/optim | XOR or linear regression trains |
| [`MODULE_8.md`](MODULE_8.md) | Hardening | Debug trace, timing, and tests stay small |

## Ground Rules

- One behavior change per patch.
- Every new op gets a correctness test.
- Every autograd rule gets a gradient test.
- Every Metal object has one obvious owner.
- No hidden global framework state.
- No optimization before a failing performance measurement.
- If an abstraction cannot be explained with a two-op example, it is probably
  too early.

