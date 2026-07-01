# MLX C Oracle Notes

`mlx-c` is the external reference for forward correctness and, later, a useful
benchmark baseline. It should not become a `comtam_lib` dependency: the
framework still owns one Metal runtime path, and tests compare that path against
MLX.

## Build

The test integration is enabled whenever `COMTAM_BUILD_TESTS=ON`. It prefers an
installed Homebrew MLX package and falls back to `mlx-c`'s `FetchContent` path
if no package is found:

```sh
cmake -S . -B build -DCOMTAM_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

## Correct Usage Pattern

Use MLX C from tests through a tiny RAII wrapper, not raw handles scattered
through test cases:

1. Create arrays with `mlx_array_new_data(..., MLX_FLOAT32)`.
2. Run an operation on a stream, preferably `mlx_default_cpu_stream_new()` for
   oracle checks.
3. Force a contiguous result before readback with `mlx_contiguous`.
4. Call `mlx_array_eval`, then `mlx_synchronize`.
5. Read `mlx_array_data_float32` while the MLX array is still alive.
6. Free every `mlx_array` and `mlx_stream`.

The current helper lives in `tests/support/mlx_oracle.h`; tensor tests include
it directly.
