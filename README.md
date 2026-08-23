# comtam

You like PyTorch? You like [micrograd](https://github.com/karpathy/micrograd)?

**comtam** is a tiny eager-mode deep learning framework in C++20 for Apple Silicon — and a self-study course for building one. Enough power to train real (small) nets. Enough complexity that you feel every abstraction before you add it. Not enough LOC to hide from your own bugs.

>[!NOTE]
>Named after [cơm tấm](https://en.wikipedia.org/wiki/C%C6%A1m_t%E1%BA%A5m): broken rice, grilled pork, a pickled thing on the side, and the quiet confidence that a good plate does not need seventeen sauces or a graph compiler.

## Why this plate exists

Most "tiny" frameworks are either adorable toys or secretly warehouses.
comtam aims at the awkward, delicious middle:

| [micrograd](https://github.com/karpathy/micrograd) | comtam | PyTorch / MLX |
| --- | --- | --- |
| scalars (and vibes) | real tensors on Metal | every dtype your ancestors invented |
| beautiful teaching toy | views, dispatch, broadcast, reduce, matmul, then autograd | ecosystem large enough to need a map |
| you finish it in a weekend | you finish it in a few weeks | you finish never |

If you want to *use* ML on a Mac, use [MLX](https://github.com/ml-explore/mlx). If you want to *own* a small tensor framework, you're in the right place.

## Recipe

- **C++20**, Apple GPU via `metal-cpp`
- **Eager**: an op submits work
- **Shared Metal buffers** first; one obvious owner per Metal object
- Correctness before speed. Speed only after a measurement that fails in public

Design notes under [`docs/`](docs/): [`ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`AUTOGRAD_DESIGN.md`](docs/AUTOGRAD_DESIGN.md), [`note/AVOID.md`](docs/note/AVOID.md), [`note/METAL_USAGE.md`](docs/note/METAL_USAGE.md).

## Build

You need:
- Apple Silicon. 
- CMake ≥ 3.24. 
- An Xcode toolchain that still remembers what Metal is (if you don't have, please follow this [tutorial](https://developer.apple.com/documentation/xcode/downloading-and-installing-additional-xcode-components#Download-and-install-the-Metal-Toolchain)).

```sh
./build.sh          # configure, build, run the demo binary
./test.sh           # build with tests and ctest
```

## Status

The eager runtime foundations through broadcasting, reductions, matmul, and primitive-surface consolidation are complete. Their original module-oriented course and grading history are frozen under [`docs/archive/course-v1/`](docs/archive/course-v1/README.md).

The active course now uses deep problem-driven chapters. The current track is [`Autograd`](docs/course/autograd/INDEX.md), beginning with [`Runtime Ownership Before Autograd`](docs/course/autograd/01_RUNTIME_OWNERSHIP.md) and [`Tensor Identity And tensor_impl`](docs/course/autograd/02_TENSOR_IDENTITY.md).

Future subjects such as neural-network modules, optimizers, sustained training, persistence, performance, and Python bindings will become their own multi-chapter tracks when the implementation reaches them.

This is a learning kitchen that still wants to ship edible food. Bring an
appetite for ownership. Leave the seventeen sauces at the door.