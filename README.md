# comtam

`comtam` is an experimental eager-mode tensor framework in C++20 for Apple
Silicon. It keeps storage, views, dispatch, and Metal execution explicit so the
runtime remains small enough to inspect and modify.

> [!WARNING]
> `comtam` is early-stage research software. Its API is unstable, platform
> coverage is narrow, and it provides no production guarantees. Use it at your
> own risk.

## Scope

The current implementation provides:

- eager execution on one Apple GPU through `metal-cpp`;
- float32 tensors backed by shared Metal buffers;
- strided views and movement operations;
- elementwise operations with broadcasting;
- reductions and matrix multiplication;
- correctness tests against MLX.

The project deliberately does not target multiple devices, multiple backends,
broad dtype coverage, Python bindings, serialization, or production serving.
Autograd is designed but not implemented.

## Build

Requirements:

- Apple Silicon;
- CMake 3.24 or newer;
- Xcode with the Metal toolchain;
- MLX when building the test suite.

Build and run the demo:

```sh
./build.sh
```

Build and run the tests with host-side sanitizers:

```sh
./test.sh
```

The sanitizers cover host C++ code, not Metal GPU memory.

## Repository map

```text
comtam/core/       Metal device, runtime, storage, and kernel dispatch
comtam/tensor/     Tensor metadata, views, and eager operations
comtam/kernels/    Metal compute kernels
tests/             Catch2 tests with MLX numerical oracles
docs/              Contributor-facing design and implementation notes
refs/              Vendored upstream samples retained for comparison
```

See [`docs/AUTOGRAD_DESIGN.md`](docs/AUTOGRAD_DESIGN.md) for the accepted
autograd direction and [`docs/note/METAL_USAGE.md`](docs/note/METAL_USAGE.md)
for Metal ownership and error-handling conventions.

## Contributing

Contributions should preserve the narrow project scope, include tests for
semantic changes, and separate correctness work from optimization. Performance
claims require a reproducible measurement and a stated baseline.

See [`AGENTS.md`](AGENTS.md) for repository-specific engineering guidance.

## License

See [`LICENSE`](LICENSE).
