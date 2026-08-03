# comtam

You like PyTorch? You like [micrograd](https://github.com/karpathy/micrograd)?
You like broken rice?

**comtam** is a tiny eager-mode deep learning framework in C++20 for Apple
Silicon — and a self-study course for building one. Enough power to train real
(small) nets. Enough complexity that you feel every abstraction before you add
it. Not enough LOC to hide from your own bugs.

> Metal-only. Float32-first. Single device. Runs *now*. No lazy graph. No fusion
> fairy. No second backend "just in case." If your Mac cannot see a GPU, lunch
> is cancelled.

Named after [cơm tấm](https://en.wikipedia.org/wiki/C%C6%A1m_t%E1%BA%A5m):
broken rice, grilled pork, a pickled thing on the side, and the quiet confidence
that a good plate does not need seventeen sauces or a graph compiler.

## Why this plate exists

Most "tiny" frameworks are either adorable toys or secretly warehouses.
comtam aims at the awkward, delicious middle:

| [micrograd](https://github.com/karpathy/micrograd) | comtam | PyTorch / "full stack" |
| --- | --- | --- |
| scalars (and vibes) | real tensors on Metal | every dtype your ancestors invented |
| beautiful teaching toy | views, dispatch, broadcast, reduce, matmul, then autograd | ecosystem large enough to need a map |
| you finish it in a weekend | you finish a *module* when the gate actually passes | you finish never |

If you want to *use* ML on a Mac, use [MLX](https://github.com/ml-explore/mlx).
If you want to *own* a small Metal runtime — storage, strides, kernels, and the
occasional existential crisis about who frees the buffer — pull up a stool.

## Course + framework (same kitchen)

Same repo. Same gates. The course is not slides around a demo — it *is* the
framework, plated module by module. Skip a gate and the next dish tastes like
undefined behavior.

```text
Metal context
  -> storage / tensor / views
  -> eager ops + broadcasting
  -> reductions, matmul
  -> reverse-mode autograd
  -> nn + SGD
  -> train something that does not lie on the test set
```

Start at [`docs/course/INDEX.md`](docs/course/INDEX.md). Printed arrays are vibes.
Oracles and epsilon are lunch.

## Shape (the recipe card)

- **C++20**, Apple GPU via `metal-cpp`
- **Eager**: an op submits work;
- **Shared Metal buffers** first; one obvious owner per Metal object
- Correctness before speed. Speed only after a measurement that fails in public

Design notes under [`docs/`](docs/): [`ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`AUTOGRAD_DESIGN.md`](docs/AUTOGRAD_DESIGN.md), [`note/AVOID.md`](docs/note/AVOID.md), [`note/METAL_USAGE.md`](docs/note/METAL_USAGE.md).

References in `refs/`: steal the arc from Magnetron, the restraint from
micrograd, and the cautionary tales from legrad (my failed previous attempt LOL).

## Build

You need:
- Apple Silicon. 
- CMake ≥ 3.24. 
- An Xcode toolchain that still remembers what Metal is (if you don't have, please follow this [tutorial](https://developer.apple.com/documentation/xcode/downloading-and-installing-additional-xcode-components)).

```sh
./build.sh          # configure, build, run the demo binary
./test.sh           # build with tests and ctest
```

Or the long way, for people who enjoy typing:

```sh
cmake -S . -B build -DCOMTAM_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

## Status

The eager runtime foundations through broadcasting, reductions, matmul, and primitive-surface consolidation are complete. Their original module-oriented course and grading history are frozen under [`docs/archive/course-v1/`](docs/archive/course-v1/README.md).

The active course now uses deep problem-driven chapters. The current track is [`Autograd`](docs/course/autograd/INDEX.md), beginning with [`Runtime Ownership Before Autograd`](docs/course/autograd/01_RUNTIME_OWNERSHIP.md) and [`Tensor Identity And tensor_impl`](docs/course/autograd/02_TENSOR_IDENTITY.md).

Future subjects such as neural-network modules, optimizers, sustained training, persistence, performance, and Python bindings will become their own multi-chapter tracks when the implementation reaches them.

This is a learning kitchen that still wants to ship edible food. Bring an
appetite for ownership. Leave the seventeen sauces at the door.

---

README flavored by a lovely AI agent from Cursor. My name is **Grok 4.5**.
I do not eat cơm tấm. I do help write about it.
