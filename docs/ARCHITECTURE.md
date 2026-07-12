# comtam Architecture

This document describes the architecture the code currently has. It should stay
close to reality so it can guide rewrites and course decisions.

`comtam` is a tiny eager-mode deep learning framework in C++20 for Apple
devices. The current implementation is Metal-only, float32-first, single-device,
and synchronous.

## Naming Convention

Types are lowercase `snake_case`, matching MLX's `array`/`Primitive` split and the STL types they interoperate with: PascalCase is reserved for the two enums (`Op`, `DType`) and would be reserved for a future polymorphic/interface type if one is ever introduced (there is none yet - no class in the codebase uses virtual dispatch).

Two namespaces split the codebase along an ownership line:

```text
comtam::           tensor-facing types: tensor, view, Op, DType
comtam::core::     runtime types: context, metal_device, storage,
                   kernel_library, command_desc and friends
```

## Current Shape

```text
context
  owns metal_device
  owns kernel_library

tensor
  owns DType
  owns view
  shares storage

storage
  owns one MTL::Buffer
  records byte size

tensor::bop (add/sub/mul/div)
  validates dtype and contiguity
  computes broadcast shape, expands both operands (stride-0, no copy)
  builds command_desc
  metal_device submits a per-element compute dispatch

tensor::matmul
  validates dtype and exact 2D shapes
  builds command_desc (no broadcasting)
  metal_device submits a threadgroup-tiled compute dispatch
```

The important boundary is:

```text
storage owns bytes.
tensor interprets bytes through a view.
view describes shape/stride/offset; it does not touch Metal.
metal_device moves bytes and launches kernels.
kernel_library maps (Op, DType) to a precompiled Metal pipeline.
```

## Source Map

```text
main.cpp
  small executable using the library

comtam/
  tensor/
    tensor.h/.cpp
      tensor metadata, host transfer, binary op and matmul entry points
    view.h/.cpp
      shape, strides, reference strides, offset, movement ops, broadcast_shape
    dtype.h
      DType and the COMTAM_DISPATCH_DTYPE macro
    op.h
      Op enum (ADD, SUB, MUL, DIV, MATMUL)

  core/
    context.h/.cpp
      top-level runtime owner (metal_device + kernel_library)
    device.h/.cpp
      Metal device, queue, allocation, copies, command submission
    storage.h/.cpp
      move-only MTL::Buffer owner
    kernel.h/.cpp
      loads a precompiled default.metallib, caches pipelines by kernel name
    command.h
      op2kernel/dtype2kernel name mapping, kernel_desc, view_desc,
      input_desc, command_desc

  kernels/
    utils.h
      ViewInfo (device-side mirror of core::view_desc) and physical_offset
    binary.metal
      add, sub, mul, div kernels, one output element per thread
    matmul.metal
      threadgroup-tiled 2D matmul kernel

  utils/
    common.h    small host helpers (error strings, file read, arr4_from_vec)
    debug.h     fmt-based formatters for view/view_desc used in debug logs
    rng.h       uniform random array generation for tests and main.cpp

  macros/
    macros.h    COMTAM_INLINE, COMTAM_FILENAME
    log.h       COMTAM_LOG_*, COMTAM_CHECK_AND_THROW, COMTAM_THROW_ERROR, COMTAM_ASSERT

tests/
  core/context.cpp
  tensor/
    view.cpp              pure view metadata
    tensor_api.cpp        storage share, from/to_vector contracts, op rejects
    forward_helper.cpp    smoke coverage for the shared comparison helper
    ops_elementwise.cpp   same-shape binary ops vs mlx-c
    ops_broadcast.cpp     broadcast binary ops vs mlx-c
    ops_matmul.cpp        matmul vs mlx-c
    movement.cpp          view movement readback vs mlx-c
  support/
    forward_compare.h   shape/value assertion helpers shared by forward tests
    mlx_oracle.h         thin mlx-c wrapper used as an independent oracle
```

## Build Flow

The root `CMakeLists.txt` builds `vendor/metal`, `comtam_lib`, the `comtam` executable, and optionally tests (which also pull in `vendor/Catch2` and `vendor/mlx-c`).

`comtam/CMakeLists.txt` copies `comtam/kernels/*.metal` and `*.h` into `build/kernels`, and - because `COMTAM_BUILD_METALLIB` defaults `ON` - compiles every kernel to `.air` with `xcrun metal` and links them into `build/kernels/default.metallib` with `xcrun metallib`. `COMTAM_KERNEL_DIR` points at that directory.

This is a real shift from earlier in the course: `kernel_library` no longer compiles Metal source at runtime. It only loads the precompiled `default.metallib` and throws if that file is missing. Runtime source compilation was the training-wheels version; a build-time `.metallib` is now the only path.

## Runtime Ownership

`core::context` is the top-level owner:

```cpp
core::context context;
auto &device = context.device();
auto &kernels = context.kernels();
```

Internally, `context` owns:

```text
std::unique_ptr<core::metal_device>
std::unique_ptr<core::kernel_library>
```

There is no global Metal manager. Keep it that way unless tests force a different ownership model.

## Device

`core::metal_device` owns:

```text
NS::SharedPtr<MTL::Device>
NS::SharedPtr<MTL::CommandQueue>
```

It is responsible for:

- creating the default Metal device and command queue
- allocating `core::storage`
- copying host arrays to storage, storage to host arrays, and storage to storage
- submitting a binary op command (`submit_bop`)
- submitting a matmul command (`submit_matmul`)

Both submit paths are synchronous: they commit a command buffer, wait with `waitUntilCompleted()`, and throw if the command buffer reports an error. This keeps lifetime simple while the framework is still proving correctness.

The two paths dispatch differently:

- `submit_bop` launches one thread per output element
  (`dispatchThreads`), sized off `view_src0.N`.
- `submit_matmul` launches a 2D grid of threadgroups
  (`dispatchThreadgroups`), tiled by the pipeline's `threadExecutionWidth`,
  and passes that width to the kernel as an explicit block size.

## Storage

`core::storage` is a dumb byte buffer:

```text
size_t size_
NS::SharedPtr<MTL::Buffer> buffer_
```

Rules:

- one `storage` owns one `MTL::Buffer`
- `storage` is move-only
- `storage` does not know `DType`
- tensors may share one `storage` through `std::shared_ptr<core::storage>`
- debug printing (`storage::print`) is not correctness evidence

This is intentionally smaller than ATen-style storage. No custom deleters, allocator cache, borrowed storage, or dtype-aware storage yet.

## Tensor

`tensor` currently stores:

```text
DType dtype_
view view_
std::shared_ptr<core::storage> storage_
```

Construction paths:

- from host data plus shape
- empty allocation from shape (contiguous, offset 0)
- header over an existing `view`
- header over existing shared storage (by shape, or by `view` directly)

Host transfer paths are dtype-dispatched through `COMTAM_DISPATCH_DTYPE`:

```text
from_vector<T>
to_vector<T>
```

The dispatch checks that the requested C++ type matches the tensor dtype. Today that means `DType::Float32 -> float`. `from_vector` still rejects non-contiguous writes; `to_vector` gathers element-by-element through `view::physical_offset`, so it already works for non-contiguous, shrunk, expanded, and reshaped views.

## View

`view` (in `comtam::`, not `comtam::core::`) currently records:

```text
shape
strides
ref_strides
offset
```

It can report:

```text
dim()
numel()
is_contiguous()
```

and produce derived views without touching storage:

```text
permute(new_axis)
transpose(a, b)
shrink(limits)
expand(new_shape)
reshape(new_shape)          -- contiguous-only, throws otherwise
broadcast_shape(lhs, rhs)   -- static, numpy-style broadcasting rules
```

The indexing rule is:

```text
physical_offset = offset + sum(index[d] * stride[d])
```

`view_desc::from_view` (in `core::command.h`) flattens a `view` into a fixed-size, GPU-uploadable struct (`shape[4]`, `strides[4]`, contiguous flag), using `-1` to pad unused rank slots. The Metal-side `ViewInfo` in `kernels/utils.h` mirrors that layout byte-for-byte, and `physical_offset` is implemented twice on purpose: once in `view.cpp` for the host, once in `utils.h` for kernels. **Rank is capped at 4** by this encoding; nothing above rank-4 tensors is supported by binary ops or matmul today.

## Operator Dispatch

### Binary ops

Binary ops live as static `tensor` methods that all funnel through
`tensor::bop`:

```cpp
static tensor add(const tensor &a, const tensor &b, core::context &ctx);
// same shape for sub, mul, div
```

The current flow is:

```text
tensor::bop
  check dtype equality
  check both inputs are contiguous
  compute broadcast_shape(a, b)
  expand both operands to that shape (stride-0 broadcast, no copy)
  allocate result tensor with the broadcast shape
  build core::command_desc from both expanded views
  metal_device::submit_bop(cmd, kernels)
  return result
```

Broadcasting is real now (unlike the earlier same-shape-only version): the kernel receives a `view_desc` per operand and indexes through `physical_offset`, so a stride-0 expanded dimension is read correctly without materializing a bigger buffer.

### Matmul

`tensor::matmul` is a separate path, not routed through `bop`:

```text
tensor::matmul
  check dtype equality
  check both operands are exactly 2D and inner dimensions match (a.cols == b.rows)
  allocate result tensor with shape {a.rows, b.cols}
  build core::command_desc from both views (no broadcasting, no expand)
  metal_device::submit_matmul(cmd, kernels)
  return result
```

`command_desc` is shared by both paths but is still deliberately narrow (two inputs, one output, one `kernel_desc`). It has no room for reductions or variadic-input ops; widen it only when a module needs that behavior.

## KernelLibrary

`core::kernel_library` loads `COMTAM_KERNEL_DIR/default.metallib` at construction and caches compute pipelines by kernel function name the first time each one is requested.

The name mapping (`op2kernel` + `dtype2kernel`, defined once in `core/command.h`) is:

```text
Op::ADD -> "add"        DType::Float32 -> "fp32"
Op::SUB -> "sub"
Op::MUL -> "mul"
Op::DIV -> "div"
Op::MATMUL -> "matmul"
```

joined as `"{op}_{dtype}"`, e.g. `"add_fp32"`, `"matmul_fp32"`. `binary.metal` exposes four float32 kernels; `matmul.metal` exposes one. All of them assume:

- one output element (or matmul output cell) per thread
- float32 data
- rank <= 4 view metadata

## Tests As Architecture

The current tests prove these architectural claims:

- `context` creates a Metal device and command queue (`tests/core/context.cpp`).
- `metal_device::copy` rejects byte-count mismatches.
- `view` builds correct row-major strides, maps linear indices to physical
  offsets, and implements permute/transpose/shrink/expand/reshape/broadcast
  correctly (`tests/tensor/view.cpp`).
- `tensor` round-trips float32 host data, rejects wrong element counts, and
  two tensor headers can safely share one `storage` with writes visible across
  headers (`tests/tensor/tensor_api.cpp`).
- `tensor::to_vector` gathers correctly through non-contiguous, shrunk,
  expanded, and reshaped views; `tensor::from_vector` still rejects
  non-contiguous writes (`tests/tensor/movement.cpp`, `tests/tensor/tensor_api.cpp`).
- Binary ops reject incompatible shapes and non-contiguous inputs before
  reaching the kernel (`tests/tensor/tensor_api.cpp`).
- Forward correctness for elementwise ops, broadcast binary ops, matmul, and
  movement ops (permute/transpose/shrink/expand/reshape) is checked against
  **mlx-c** as an independent oracle, not a hand-rolled CPU reference
  (`tests/tensor/ops_*.cpp`, `tests/tensor/movement.cpp`,
  `tests/support/mlx_oracle.h`, `tests/support/forward_compare.h`).

If a claim matters architecturally, it should eventually have a test like this.

## Not In The Architecture Yet

These are intentionally absent:

- Python bindings
- multiple backends
- dynamic backend loading
- allocator caches
- many dtypes
- serialization
- async command scheduling
- lazy graphs or fusion
- reductions (sum/mean/max/...) - `reduction.metal` at the repo root is
  scratch/WIP, untracked, unbuilt, and not wired into `comtam/kernels/` or any
  CMake target; treat it as a draft, not part of the current architecture
- autograd
- `nn` modules and optimizers
- profiling counters

Do not add them for symmetry. Add them only when a course module or failing
test earns the complexity.

## Near-Term Pressure Points

The next likely rewrites, following Module 5's remaining scope:

1. Finish reductions (sum first): decide the kernel contract (threadgroup tree vs. simple strided reduce), and how `command_desc`/`view_desc` need to grow to describe a reduced-away axis.
2. Decide whether `command_desc` should grow a variant/tag for reduction and future ops, or whether reductions deserve their own narrow descriptor the way matmul got its own `submit_matmul` instead of overloading `bop`.
3. Revisit the rank-4 cap in `view_desc`/`ViewInfo` if any planned op needs higher-rank tensors.
4. Decide whether binary ops and matmul should stay as static `tensor` methods or become free functions once the public API keeps growing.

## Evolution Rule

Every new subsystem must answer:

```text
What current code became simpler or more correct because this exists?
```

If the honest answer is only "future flexibility", wait.
