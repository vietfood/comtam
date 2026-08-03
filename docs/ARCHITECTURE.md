# comtam Architecture

This document describes the architecture implemented by the current source. It
is living guidance: when a deliberate, tested source design supersedes an older
course choice, update this document and the active module contract together.

The accepted but not yet implemented `tensor_impl`, tensor-bound runtime, default runtime, and reverse-mode ownership design lives in [`AUTOGRAD_DESIGN.md`](AUTOGRAD_DESIGN.md). This document continues to describe the current explicit-context implementation until the corresponding source and tests land.

`comtam` is a tiny eager-mode deep-learning framework in C++20 for Apple
devices. It is currently Metal-only, float32-only, single-device, and
synchronous.

## Naming And Ownership

Types use lowercase `snake_case`; `Op`, `OpVariant`, and `DType` are enums.
Namespaces follow the tensor/runtime ownership boundary:

```text
comtam::        tensor, view, Op, OpVariant, DType
comtam::core::  context, metal_device, storage, kernel_library,
                command_desc and GPU descriptors
```

The ownership tree is:

```text
core::context
  owns one core::metal_device
    owns one MTL::Device
    owns one MTL::CommandQueue
  owns one core::kernel_library
    caches MTL::ComputePipelineState objects

tensor
  owns DType and view metadata
  shares core::storage

core::storage
  owns exactly one MTL::Buffer
  records its byte size
```

There is no hidden global framework state. Every operation receives an explicit
`core::context`.

## Source Map

```text
comtam/tensor/
  tensor.h/.cpp   tensor API, compositions, dispatch preparation
  view.h/.cpp     shape, strides, offset, movement and broadcast metadata
  checks.h        side-effect-free operation preflight
  op.h            semantic dispatch operations and layout variants
  dtype.h         float32 dtype dispatch

comtam/core/
  context.*       top-level runtime owner
  device.*        allocation, copies, command encoding and synchronous submit
  storage.*       move-only shared Metal-buffer owner
  kernel.*        precompiled metallib loading and pipeline cache
  command.h       kernel naming, fixed-rank GPU view and command descriptors

comtam/kernels/
  binary.metal    add and multiply
  unary.metal     negation and reciprocal
  reduction.metal full/axis sum and maximum
  matmul.metal    contiguous tiled and general strided matmul
  utils.h         Metal ViewInfo and physical-offset mapping

tests/
  tensor/checks.cpp          metadata-preflight tests without Metal dispatch
  tensor/view.cpp            pure view and broadcast-shape tests
  tensor/tensor_api.cpp      tensor/storage contracts and public rejection paths
  tensor/movement.cpp        public view operations against MLX-C
  tensor/ops_elementwise.cpp unary and same-shape elementwise oracles
  tensor/ops_broadcast.cpp   broadcast and non-contiguous binary oracles
  tensor/ops_reduce.cpp      full/axis reduction and mean oracles
  tensor/ops_matmul.cpp      contiguous/strided matmul oracles
  tensor/ops_numerics.cpp    exceptional float32 classifications and tolerances
  support/mlx_oracle.h       MLX-C CPU-stream oracle adapter
```

## Data Model

### Storage

`core::storage` owns one valid shared-mode `MTL::Buffer`. It is move-only and
does not know dtype, shape, strides, or offset. Zero-byte allocation remains
unsupported; Module 5 operation preflight rejects zero extents before output
allocation or dispatch.

Tensors share storage through `std::shared_ptr<core::storage>`. Multiple tensor
headers may therefore describe different logical views of the same bytes.

### View

`view` records:

```text
shape
strides
ref_strides
offset
```

Movement operations derive new metadata without moving storage:

```text
permute, transpose, shrink, expand, reshape
```

The logical-to-physical rule is:

```text
physical_offset = offset + sum(index[d] * stride[d])
```

`reshape` is currently contiguous-only. `expand` represents broadcasting with
stride-zero dimensions. `view::broadcast_shape` implements right-aligned
NumPy-style shape inference, including rank-0 and symmetric zero-extent shape
math; tensor operations still reject zero extents in the current runtime.

### Tensor

`tensor` stores:

```text
DType dtype_
view view_
std::shared_ptr<core::storage> storage_
```

Rank-0 tensors are ordinary tensors with shape `()`, one logical element, and a
normal float32 buffer. There is no separate scalar command or scalar kernel
family.

Host writes require contiguous tensors. Host readback walks
`view::physical_offset`, so transpose, shrink, expand, and other non-contiguous
views are read in logical order.

## GPU Metadata And Validation

`core::view_desc` is the fixed GPU representation of a view:

```text
N
shape[4]
strides[4]
offset
contiguous
```

Metal `ViewInfo` mirrors this layout. Rank is capped at four because of these
fixed arrays. Unused slots are padded during conversion.

Metadata validation lives in `comtam/tensor/checks.h` and runs before output
allocation or dispatch:

- `check_unary`: supported dtype, rank <= 4, positive extents.
- `check_binary`: matching supported dtype, rank/extents, and broadcast shape.
- `check_reduce_full` / `check_reduce_axis`: tensor checks, contiguous input,
  non-negative valid axis, and keepdim output shape.
- `check_matmul`: matching supported dtype, positive extents, rank exactly two,
  and matching inner dimensions.

Reductions intentionally remain contiguous-only. Binary and unary kernels read
arbitrary supported views through physical offsets. Matmul selects a fast
zero-offset contiguous path or a general strided path.

## Operator Surfaces

Public names, semantic graph nodes, and Metal functions are different surfaces.

### Initially differentiable semantic primitives

Module 6 will record and differentiate:

```text
add, mul, neg, recip, sum, matmul
public movement operations
```

### Forward-only semantic primitive

`max` is independently dispatched and tested, but its gradient is deferred.
Until a tie policy and gradient test exist, Module 6 must reject recording
through `max`. Composed `min` inherits that restriction.

### Public compositions

```text
sub(a, b)        = add(a, neg(b))
div(a, b)        = mul(a, recip(b))
mean(a)          = mul(sum(a), scalar(1.0F / a.numel()))
mean(a, axis, k) = mul(sum(a, axis, k),
                       scalar(1.0F / a.shape[axis]))
min(a, ...)      = neg(max(neg(a), ...))
```

These public names do not have dedicated `Op` values, semantic nodes, or Metal
entry points. Their primitive calls form the future autograd graph. Framework
constants do not require gradients and therefore do not add tape nodes.

### Physical Metal entry points

| Semantic operation | Physical entry point(s) |
| --- | --- |
| `add` | `add_fp32` |
| `mul` | `mul_fp32` |
| `neg` | `neg_fp32` |
| `recip` | `recip_fp32` |
| `sum` | `reduce_sum_full_fp32`, `reduce_sum_axis_fp32` |
| `max` | `reduce_max_full_fp32`, `reduce_max_axis_fp32` |
| `matmul` | `matmul_contiguous_fp32`, `matmul_strided_fp32` |

Layout and reduction variants are backend details; they do not create more
semantic operations or future gradient rules.

## Dispatch Paths

### Binary primitives

`tensor::bop` performs binary preflight, computes the final broadcast shape,
and creates raw internal stride-zero view aliases over the original storages.
It deliberately does not call public `tensor::expand`.

```text
check_binary(original a, original b)
  -> internal a.view.expand(result shape)
  -> internal b.view.expand(result shape)
  -> allocate contiguous output
  -> submit_bop(add or mul)
```

This is the future tape boundary: semantic binary nodes refer to the original
operands. Internal broadcast aliases are submission metadata and never tape
nodes. A user-authored `tensor::expand` remains a real movement operation.

### Unary primitives

`neg` and `recip` validate one input, allocate a same-shape contiguous output,
and submit one thread per output element. The kernel reads the input using its
shape, strides, and offset, including transposed and nonzero-offset views.

### Reductions

`sum` and forward-only `max` share a reduction submission path with `FULL` and
`AXIS` variants. A full reduction returns a rank-0 tensor. Axis reduction either
removes the axis or retains it with extent one according to `keepdim`.

The kernels use one threadgroup for a full result or one threadgroup per axis
output element. `mean` and `min` are tensor-layer compositions, not reduction
kernels.

### Matmul

`matmul` accepts exactly two rank-2 tensors and returns a new contiguous
`{a.rows, b.cols}` tensor.

- Zero-offset contiguous operands select a tiled `16 x 16` threadgroup-memory
  kernel.
- Any supported strided or offset operand selects the general kernel, which
  computes one output cell per thread and reads both inputs through
  `physical_offset`.

The physical choice implements one semantic `matmul` operation.

## Runtime Submission

`core::command_desc` carries one `kernel_desc`, two optional tensor inputs, one
output storage, and small operation-specific metadata such as a reduction axis.
`kernel_desc` combines `Op`, `DType`, and `OpVariant` to form the physical Metal
function name.

`core::metal_device` has narrow submit methods for binary, unary, reduction,
and matmul work. Each command creates a command buffer and compute encoder,
dispatches synchronously, waits for completion, and reports Metal errors before
returning.

`core::kernel_library` loads the build-produced `default.metallib` and lazily
caches compute pipelines by the complete physical function name.

## Numerical Policy

The supported arithmetic dtype is float32. MLX-C running on its CPU stream is
the independent numerical oracle.

### Finite comparisons

- The shared approximate forward matcher uses absolute tolerance `1e-5`.
- Finite reciprocal and composed division use one fixed absolute-plus-relative
  tolerance: `abs_eps = 1e-5`, `rel_eps = 1e-5`.
- Reductions use absolute tolerance `1e-4` because their parallel summation
  order can differ from MLX-C.
- Shape, metadata, movement indices, and exact host-transfer claims use exact
  comparisons where appropriate.

Tolerances apply only to finite values. NaN and infinity are classified rather
than epsilon-compared.

### Exceptional values

`neg` preserves IEEE-style sign/classification behavior:

```text
neg(+0)   -> -0
neg(-0)   -> +0
neg(+inf) -> -inf
neg(-inf) -> +inf
neg(NaN)  -> NaN
```

`recip` is the float32 operation `1.0F / x`:

```text
recip(+0)   -> +inf
recip(-0)   -> -inf
recip(+inf) -> +0
recip(-inf) -> -0
recip(NaN)  -> NaN
```

Public division is `mul(a, recip(b))`. Its promised classification follows
IEEE-style sign XOR for signed zero and infinity:

| Numerator / denominator | Classification |
| --- | --- |
| finite nonzero / finite nonzero | finite, within the fixed abs+rel tolerance |
| finite nonzero or signed infinity / signed zero | signed infinity |
| signed zero / signed zero | NaN |
| signed zero / finite nonzero | signed zero |
| signed zero or finite nonzero / signed infinity | signed zero |
| signed infinity / finite nonzero | signed infinity |
| signed infinity / signed infinity | NaN |
| either arithmetic input is NaN | NaN |

Because `div` is composed, finite rounding may differ from one hardware divide
instruction; that is why its finite contract uses both absolute and relative
tolerance.

`mean` computes a positive finite host-side inverse count and multiplies it by
the sum result. NaN and infinity propagation therefore follows `sum` and `mul`.
The supported positive-extent/rank-4 scope prevents a zero divisor. Exceptional
`max`/`min` behavior is not yet promised; their current oracle coverage uses
finite inputs.

### Metal compiler policy

Metal sources are compiled by CMake with:

```text
xcrun -sdk macosx metal -c <source> -o <air>
```

The build does not request `-ffast-math`, finite-only assumptions, or
`-fno-signed-zeros`. The exceptional-value tests are executable enforcement of
the signed-zero/NaN/infinity contract on the supported toolchain. Any future
compiler-flag change must keep those tests green or explicitly revise the
public numerical contract.

Denormal handling, overflow guarantees beyond ordinary float32 behavior, and
cross-toolchain bitwise reproducibility are not currently specified.

## Build Flow

The root CMake project builds `metal-cpp`, `comtam_lib`, the `comtam`
executable, and optionally Catch2 tests with MLX-C. `comtam/CMakeLists.txt`
compiles each `.metal` file to `.air`, links them into
`build/kernels/default.metallib`, and exposes that directory through
`COMTAM_KERNEL_DIR`.

There is no runtime Metal source compilation.

## Tests As Architecture

Current tests establish:

- storage ownership, sharing, bounds, and host transfer contracts;
- rank-0 through rank-4 view metadata and public rejection above rank four;
- right-aligned broadcasting, including scalars and zero-extent shape math;
- zero-extent operation rejection before dispatch;
- unary and binary primitives on contiguous, transposed, and offset views;
- composed `sub` and `div` on scalar, asymmetric, and non-contiguous inputs;
- full/axis `sum`, `mean`, finite `max`, and finite `min` against MLX-C;
- contiguous tiled and strided matmul against MLX-C;
- the currently enumerated negation, reciprocal, and division exceptional-value
  cases.

Printed arrays are debugging only; correctness evidence comes from Catch2
assertions and independent MLX-C comparisons.

## Deferred Architecture

The following remain intentionally deferred:

- autograd and gradient storage;
- a gradient/tie policy for `max` and composed `min`;
- non-contiguous reductions;
- zero-sized storage and empty-tensor arithmetic;
- additional dtypes, dtype promotion, and multiple devices/backends;
- asynchronous scheduling, allocator caches, fusion, and lazy graphs;
- Python bindings, serialization, `nn` modules, and optimizers.

The immediate next architecture step is the [`Autograd Track`](course/autograd/INDEX.md): implement the accepted design over the frozen semantic surface without allowing internal broadcast aliases or constant-only branches to become graph nodes.

## Evolution Rule

Every new subsystem must answer:

```text
What current code became simpler, more correct, or more educational because
this exists?
```

If the honest answer is only “future flexibility,” wait.
