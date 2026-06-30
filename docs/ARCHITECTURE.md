# comtam Architecture

This document describes the architecture the code currently has. It should stay
close to reality so it can guide rewrites and course decisions.

`comtam` is a tiny eager-mode deep learning framework in C++20 for Apple
devices. The current implementation is Metal-only, float32-first, single-device,
and synchronous.

## Current Shape

```text
Context
  owns Device
  owns KernelLibrary

Tensor
  owns DType
  owns View
  shares Storage

Storage
  owns one MTL::Buffer
  records byte size

Tensor op
  validates enough for the current module
  allocates result storage
  builds Command
  Device submits Command to Metal
```

The important boundary is:

```text
Storage owns bytes.
Tensor interprets bytes.
Device moves bytes and launches kernels.
KernelLibrary maps ops to Metal pipelines.
```

## Source Map

```text
main.cpp
  small executable using the library

comtam/
  tensor.h/.cpp
    Tensor metadata, host transfer, binary op entry points

comtam/core/
  context.h/.cpp
    top-level runtime owner
  device.h/.cpp
    Metal device, queue, allocation, copies, command submission
  storage.h/.cpp
    move-only MTL::Buffer owner
  view.h
    shape, strides, reference strides, offset
  dtype.h
    DType and dtype dispatch
  ops.h
    Op enum, kernel-name lookup, binary Command
  kernel.h/.cpp
    Metal source compilation and pipeline cache

comtam/kernels/
  binary.metal
    add, sub, mul, div kernels

tests/
  core/context.cpp
  tensor/v0.cpp
```

## Build Flow

The root `CMakeLists.txt` builds `vendor/metal`, `comtam_lib`, the `comtam`
executable, and optionally tests.

`comtam/CMakeLists.txt` copies `comtam/kernels/*.metal` into
`build/kernels`, defines `COMTAM_KERNEL_DIR`, and can build
`default.metallib`. The current `KernelLibrary` path compiles all copied Metal
source files at runtime from `COMTAM_KERNEL_DIR`.

This is good enough for now. Loading a precompiled `.metallib` can wait until
runtime compilation becomes a real pain.

## Runtime Ownership

`core::Context` is the top-level owner:

```cpp
core::Context context;
auto &device = context.device();
auto &kernels = context.kernels();
```

Internally, `Context` owns:

```text
std::unique_ptr<Device>
std::unique_ptr<KernelLibrary>
```

There is no global Metal manager. Keep it that way unless tests force a
different ownership model.

## Device

`core::Device` owns:

```text
NS::SharedPtr<MTL::Device>
NS::SharedPtr<MTL::CommandQueue>
```

It is responsible for:

- creating the default Metal device and command queue
- allocating `Storage`
- copying host arrays to storage
- copying storage to host arrays
- copying storage to storage
- submitting a binary `Command`

`Device::submit` is synchronous. It commits a command buffer, waits with
`waitUntilCompleted()`, and throws if the command buffer reports an error. This
keeps lifetime simple while the framework is still proving correctness.

## Storage

`core::Storage` is a dumb byte buffer:

```text
size_t size_
NS::SharedPtr<MTL::Buffer> buffer_
```

Rules:

- one `Storage` owns one `MTL::Buffer`
- `Storage` is move-only
- `Storage` does not know `DType`
- tensors may share one `Storage` through `std::shared_ptr<Storage>`
- debug printing is not correctness evidence

This is intentionally smaller than ATen-style storage. No custom deleters,
allocator cache, borrowed storage, or dtype-aware storage yet.

## Tensor

`Tensor` currently stores:

```text
core::DType dtype_
core::View view_
std::shared_ptr<core::Storage> storage_
```

Construction paths:

- from host data plus shape
- empty allocation from shape
- header over existing shared storage

Host transfer paths are dtype-dispatched:

```text
from_vector<T>
to_vector<T>
```

The dispatch checks that the requested C++ type matches the tensor dtype. Today
that means `DType::Float32 -> float`.

Current limitation: `to_vector` assumes the tensor view can be read as a
contiguous buffer. Module 2 is where view-aware readback earns its place.

## View

`core::View` currently records:

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

The indexing rule this is growing toward is:

```text
physical_offset = offset + sum(index[d] * stride[d])
```

The current binary kernels do not consume view metadata yet. That is the next
architectural pressure point.

## Operator Dispatch

Binary ops currently live as static `Tensor` methods:

```cpp
Tensor Tensor::add(const Tensor &a, const Tensor &b, core::Device &device,
                   core::KernelLibrary &kernels);
```

The current flow is:

```text
Tensor::add
  allocate result Tensor with a's shape
  build core::Command
  Device::submit(command, kernels)
  return result
```

`core::Command` is deliberately narrow:

```text
Op op
Storage* a
Storage* b
Storage* out
size_t elements
```

This is enough for same-shape contiguous binary ops. It is not enough for
broadcasting, reductions, matmul, or non-contiguous views. Widen it only when a
module needs that behavior.

## KernelLibrary

`core::KernelLibrary` compiles Metal source from `COMTAM_KERNEL_DIR` and caches
compute pipelines by kernel function name.

The mapping is:

```text
Op::ADD -> "add"
Op::SUB -> "sub"
Op::MUL -> "mul"
Op::DIV -> "div"
```

`binary.metal` currently exposes those four float32 kernels. The kernels assume:

- contiguous buffers
- one output element per thread
- float32 data

Module 3 should pin down the kernel contract before adding more ops.

## Tests As Architecture

The current tests prove these architectural claims:

- `Context` creates a Metal device and command queue.
- `Device::copy` rejects byte-count mismatches.
- `Tensor` round-trips float32 host data.
- `Tensor::from_vector` rejects wrong element counts.
- two tensor headers can share one `Storage` safely.
- writes through shared storage are visible through another tensor header.
- `Tensor::add` matches a CPU oracle for same-shape vectors.

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
- autograd
- `nn` modules and optimizers
- profiling counters

Do not add them for symmetry. Add them only when a course module or failing test
earns the complexity.

## Near-Term Pressure Points

The next likely rewrites are:

1. Make `Tensor::to_vector` and future kernels respect `View` strides and
   offsets.
2. Move binary op validation out of wishful thinking and into explicit shape,
   dtype, and contiguity checks.
3. Decide whether binary ops should stay as static `Tensor` methods or become
   free functions once the public API starts to grow.
4. Extend `Command` only when broadcasting, reductions, or matmul need more
   metadata.

## Evolution Rule

Every new subsystem must answer:

```text
What current code became simpler or more correct because this exists?
```

If the honest answer is only "future flexibility", wait.
