# comtam Architecture

This is the proposed architecture for `comtam`. It is intentionally temporary.
The design should evolve as the framework earns new requirements.

The current target is a tiny eager-mode deep learning framework in C++20 with
Metal as the only external runtime dependency.

## Thesis

`comtam` should borrow Magnetron's architectural discipline, not Magnetron's
full scope.

Magnetron's useful shape is:

```text
Context
  -> Tensor and Storage
  -> Device command dispatch
  -> Autograd metadata
  -> Modules and optimizers
```

`comtam` keeps that shape, but narrows it:

```text
one backend: Metal
one dtype first: float32
one device first: default Apple GPU
eager execution first
runtime clarity before performance
```

## Proposed Layers

```text
main.cpp
  uses public comtam API

comtam/core/
  context
  device
  kernel
  storage
  tensor
  op
  autograd

comtam/nn/
  parameter
  module
  linear
  loss
  optim

comtam/kernels/
  vadd.metal
  elementwise.metal
  reduce.metal
  matmul.metal
```

This layout is not sacred. It is a starting point.

## Runtime Ownership

The top-level owner is `Context`.

```text
Context
  owns Device
  owns KernelLibrary
  later owns Allocator
```

There should be no global `MetalManager`.

Preferred use:

```cpp
comtam::Context ctx;
auto& device = ctx.device();
auto& kernels = ctx.kernels();
```

This keeps tests, lifetime, and future experiments explicit.

## Device

`Device` is the Metal execution owner.

Initial responsibilities:

- own `MTL::Device`
- own `MTL::CommandQueue`
- allocate `MTL::Buffer`
- submit compute commands

Not initial responsibilities:

- allocator cache
- profiling counters
- multiple GPU selection
- async graph execution

Future direction:

```cpp
class Device {
 public:
  Storage allocate(size_t bytes);
  void submit(const Command& command);
};
```

## KernelLibrary

`KernelLibrary` manages Metal compute pipelines.

Initial responsibilities:

- load `.metal` source files from `build/kernels`
- compile a library at runtime, or load `default.metallib` when available
- create and cache `MTL::ComputePipelineState` by kernel name

Suggested shape:

```cpp
class KernelLibrary {
 public:
  KernelLibrary(Device& device, std::filesystem::path kernel_dir);
  MTL::ComputePipelineState* pipeline(std::string_view name);
};
```

This is deliberately smaller than Magnetron's backend registry. `comtam` does
not need backend modules, ABI cookies, or dynamic device discovery.

## Storage

`Storage` owns raw device memory.

Initial shape:

```text
Storage
  MTL::Buffer*
  size_bytes
```

Rules:

- one `Storage` owns one buffer
- tensors may share storage
- views do not allocate
- no borrowed storage in the first version
- no custom deleter system in the first version

## Tensor

`Tensor` is storage plus interpretation.

Initial metadata:

```text
dtype
shape
strides
storage_offset
storage
requires_grad
grad_node
```

The core formula:

```text
physical_offset = storage_offset + sum(index[d] * stride[d])
```

Views are tensor headers that share storage. This mirrors Magnetron's good
design choice without copying its C reference-counted implementation.

## Operator Dispatch

Public tensor ops should be small.

```text
add(a, b)
  validate
  broadcast if needed
  allocate result
  record autograd metadata if needed
  submit Command
  return result
```

The important Magnetron-inspired idea:

```text
high-level op -> uniform command -> device submit
```

Initial command shape:

```cpp
enum class OpCode {
  Fill,
  Add,
  Sub,
  Mul,
  Div,
  Neg,
  Relu,
  Sum,
  Mean,
  Matmul,
};

struct Command {
  OpCode op;
  std::span<Tensor*> inputs;
  std::span<Tensor*> outputs;
};
```

This can change. The principle should not: do not hide kernel launches inside
unrelated helper functions.

## Autograd

Autograd should be dynamic and eager.

Each differentiable output stores enough information to compute local gradients:

```text
GradNode
  op
  inputs
  attrs
  grad
```

Backward:

1. require scalar root
2. topologically sort the dynamic graph
3. seed root gradient with one
4. traverse reverse topological order
5. call local backward rules
6. accumulate gradients

Do not introduce a separate tape abstraction until storing nodes on tensors is
clearly insufficient.

## Initial Build Flow

The current CMake direction is:

```text
vendor/metal
  builds metal_cpp shim

comtam
  builds executable from comtam/main.cpp
  copies comtam/kernels/*.metal to build/kernels
  optionally compiles default.metallib
```

Runtime source compilation is acceptable for the first GPU experiment. Offline
`.metallib` compilation can become the default after the toolchain and workflow
are stable.

## Module 0 Target Architecture

Module 0 should end with:

```text
comtam/
  main.cpp
  core/
    context.h/.cpp
    device.h/.cpp
    kernel.h/.cpp
  kernels/
    vadd.metal
```

The first real proof:

```text
host vectors -> Metal buffers -> vadd kernel -> host readback
```

No tensor class is required for this proof.

## Module 1 Target Architecture

Module 1 should add:

```text
storage.h/.cpp
tensor.h/.cpp
```

The proof:

```text
Tensor::from_vector
Tensor::to_vector
Tensor add on GPU
```

## Things Not In The Architecture Yet

These are intentionally excluded:

- singleton manager
- ATen-like allocator/deleter contexts
- backend registry
- multiple devices
- Python bindings
- lazy graph
- kernel fusion
- serialization
- profiling counters

See [`AVOID.md`](AVOID.md) for the reasoning.

## Evolution Rule

Every new subsystem must answer:

```text
What current code became simpler because this exists?
```

If the answer is "future flexibility", wait.
