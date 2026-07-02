# Metal 4 Study Guide

This note is a map for studying the `refs/metal_example/metal_4/` and
`refs/metal_example/metal_4_2/` sample apps, cross-referenced against the
Metal 4 headers `comtam` actually links against in
`vendor/metal/metal-cpp/Metal/MTL4*.hpp`, `MTLTensor.hpp`, and
`MTLResidencySet.hpp`.

It complements [`MAGENETRON.md`](MAGENETRON.md) and [`MLX.md`](MLX.md), but it
is a different kind of reference. Magnetron and MLX are whole *frameworks*
(runtime shape, kernel-dispatch shape). This note is about the *GPU API
surface underneath* any framework choice - the thing `comtam/core/device.cpp`
already talks to today, and the thing it would talk to differently if it
adopted Metal 4's newer command-submission and tensor-shader features.

Treat this like [`MLX_REDUCTION.md`](MLX_REDUCTION.md): a focused technical
reference to borrow *individual patterns* from once a measurement or a course
module earns them, not a rewrite target.

---

## What Metal 4 Is

"Metal 4" is a marketing umbrella over three genuinely separate upgrades. Do
not treat them as one feature - they have different relevance to `comtam`.

1. **A new command-submission API** (`MTL4CommandQueue`, `MTL4CommandBuffer`,
   `MTL4CommandAllocator`, `MTL4ArgumentTable`, `MTL4Compiler`). This replaces
   the classic "queue makes a buffer, buffer makes an encoder, encoder does
   ad-hoc `setBuffer:atIndex:` calls, `commit` + `waitUntilCompleted`" model
   with an explicit split between *recording memory* (allocator), *argument
   binding* (argument table), and *pipeline compilation* (compiler). The goal
   is lower CPU overhead per encoded command and more control over when work
   actually blocks.
2. **A native tensor resource type** (`MTL::Tensor` / `id<MTLTensor>` in
   `MTLTensor.hpp`) - a GPU resource that carries rank, per-dimension extents,
   and a `MTLTensorDataType`, instead of being a raw `MTL::Buffer` that only
   host-side code knows how to interpret.
3. **New Metal Shading Language (MSL 4) tensor types and cooperative-tensor
   matmul operations** (`mpp::tensor_ops`, `tensor<...>`, `matmul2d<...>`),
   which let a *kernel itself* express tiling, slicing, and matrix multiply
   against the GPU's native tensor/neural-accelerator hardware, instead of a
   hand-written threadgroup-memory GEMM.

Plus one more piece that rides on top of (1) and (2):
`MTL4MachineLearningCommandEncoder`, which dispatches a whole CoreML/
`.mtlpackage` model inline in your own Metal command buffer, on the GPU
timeline, with no CPU round trip between "GPU finishes matmul" and "GPU runs
the next pass."

These four pieces are separable. `refs/metal_example/metal_4_2` proves this:
it uses the new MTL4 command/allocator/argument-table plumbing together with
an entirely classic `id<MTLComputePipelineState>` (built the old way, not
through `MTL4Compiler`) and a hand-written `.metal` kernel that happens to use
the new MSL 4 tensor types inside its kernel body.

---

## Architecture In One Diagram

Classic Metal, which is what `comtam/core/device.cpp` uses today:

```text
MTL::CommandQueue (owns internal buffer pool)
        |
        v
command_queue_->commandBuffer()      -- implicit allocation
        |
        v
command_buffer->computeCommandEncoder()
  encoder->setComputePipelineState(pipeline)
  encoder->setBuffer(buf, offset, index)     -- one call per binding
  encoder->setBytes(&view, size, index)
  encoder->dispatchThreads(threads, threads_per_group)
  encoder->endEncoding()
        |
        v
command_buffer->commit()
command_buffer->waitUntilCompleted()   -- synchronous, always
```

Metal 4, as used by both sample apps:

```text
MTL4::CommandAllocator            -- separate, resettable recording memory
        |
device->newCommandBuffer()        -- buffer is cheap/reusable, allocator holds the bytes
        |
        v
commandBuffer->beginCommandBuffer(allocator)
        |
        +-- computeCommandEncoder()              (metal_4_2: matmul kernel)
        |     encoder->setComputePipelineState(state)
        |     encoder->setArgumentTable(table)    -- bindings set on the table, not the encoder
        |     encoder->dispatchThreadgroups(...)
        |
        +-- machineLearningCommandEncoder()       (metal_4: CoreML matmul)
        |     encoder->setArgumentTable(table)
        |     encoder->setPipelineState(mlPipelineState)
        |     encoder->dispatchNetworkWithIntermediatesHeap(heap)
        |
        v
commandBuffer->endCommandBuffer()
queue->commit(&commandBuffer, 1)     -- does not block
queue->signalEvent(sharedEvent, value)
sharedEvent->waitUntilSignaledValue(value, timeoutMS)   -- explicit CPU/GPU sync point
```

The important structural shift: **recording memory, argument binding, and
pipeline compilation are now three separate long-lived objects** you create
once and reuse, instead of being folded into "queue" and "encoder." That
reuse is the entire performance argument for Metal 4's command API - it is
not about doing new GPU work, it is about doing the same GPU work with less
CPU-side bookkeeping per dispatch.

---

## The Two Sample Apps, Mapped

### `metal_4/` - machine learning pass (CoreML model on the GPU timeline)

Read in this order:

1. `Matrix multiplier/MLMatrixMultiplier.m` (`init`) - creates
   `MTL4CommandQueue`, loads a `.mtlpackage` as an `id<MTLLibrary>`, creates an
   `MTL4Compiler`, an `MTLSharedEvent`, an `MTL4CommandBuffer`, and an
   `MTL4CommandAllocator`. All of these are created once and reused - the
   README calls this out explicitly as the pattern long-running apps should
   follow.
2. `Matrix multiplier/MLMatrixMultiplier+PipelineCompilation.m` - reads
   `MTLFunctionReflection` off the library to find tensor-typed bindings,
   builds an `MTL4MachineLearningPipelineDescriptor` with explicit
   `MTLTensorExtents` per input, and compiles it via
   `compiler->newMachineLearningPipelineState(descriptor, error)`. Note that
   *compilation is a `Compiler` method, not a `Device` method* - this is true
   for every MTL4 pipeline type (compute, render, ML).
3. `Matrix multiplier/MLMatrixMultiplier+TensorSetup.m` - walks
   `pipelineState.reflection.bindings`, creates one `id<MTLTensor>` per
   binding via `MTLTensorDescriptor`, and copies matrix data in with
   `getBytes:strides:fromSliceOrigin:sliceDimensions:` /
   `replaceSliceOrigin:sliceDimensions:withBytes:strides:` (`Matrix.m`).
4. `Matrix multiplier/MLMatrixMultiplier.m`
   (`encodeAndRunModelInference`) - the whole dispatch:

```223:256:refs/metal_example/metal_4/Matrix multiplier/MLMatrixMultiplier.m
- (void)encodeAndRunModelInference {
    [commandBuffer beginCommandBufferWithAllocator:commandAllocator];

    id<MTL4MachineLearningCommandEncoder> encoder;
    encoder = [commandBuffer machineLearningCommandEncoder];

    [encoder setArgumentTable:argumentTable];
    [encoder setPipelineState:pipelineState];
    [encoder dispatchNetworkWithIntermediatesHeap:intermediatesHeap];
    [encoder endEncoding];

    [commandBuffer endCommandBuffer];

    [commandQueue commit:&commandBuffer count:1];

    // Add a command to the queue that increments the shared event's value.
    uint64_t signalValue = sharedEvent.signaledValue + 1;
    [commandQueue signalEvent:sharedEvent value:signalValue];
    ...
    BOOL success = [sharedEvent waitUntilSignaledValue:signalValue
                                             timeoutMS:kMLPassTimeoutMilliseconds];
```

Notice there is no `waitUntilCompleted()` anywhere in this sample. MTL4
command buffers do not expose it; a `MTLSharedEvent` signal/wait pair is the
only synchronization primitive shown. That is a deliberate API change, not an
oversight - see Core Idea 5 below.

### `metal_4_2/` - inline tensor ops in a compute kernel (cooperative-tensor matmul)

Read in this order:

1. `Matrix multiplier/Shaders.metal` - the entire point of this sample:

```46:87:refs/metal_example/metal_4_2/Matrix multiplier/Shaders.metal
kernel void
matrix_multiplication_kernel(uint2 threadgroupIdentifier [[threadgroup_position_in_grid]],
                             tensor<device half, dextents<int, 2>> sourceA,
                             tensor<device half, dextents<int, 2>> sourceB,
                             tensor<device half, dextents<int, 2>> destination)
{
    const int TileSize = 64;
    int tileOriginX = TileSize * threadgroupIdentifier.x;
    int tileOriginY = TileSize * threadgroupIdentifier.y;

    auto sliceA = sourceA.slice<dynamic_extent, TileSize>(0, tileOriginY);
    auto sliceB = sourceB.slice<TileSize, dynamic_extent>(tileOriginX, 0);
    auto destinationSlice = destination.slice<TileSize, TileSize>(tileOriginX, tileOriginY);

    constexpr auto descriptor = matmul2d_descriptor(TileSize, TileSize, dynamic_length_v<int>);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    auto cooperativeTensor = operation.get_destination_cooperative_tensor<
        decltype(sliceA), decltype(sliceB), half>();
    operation.run(sliceA, sliceB, cooperativeTensor);

    for (int element = 0; element < cooperativeTensor.get_capacity(); element++) {
        auto value = cooperativeTensor[element];
        cooperativeTensor[element] = relu(value);
    }
    cooperativeTensor.store(destinationSlice);
}
```

   Compare this mentally to a from-scratch tiled matmul (threadgroup-memory
   tiles, manual SIMD-group indexing, manual accumulate loop over the K
   dimension). This kernel has *none of that*: `matmul2d` and the cooperative
   tensor own the tiling, the K-loop, and the register/threadgroup placement
   decision. `execution_simdgroups<4>` is the only tuning knob exposed.
2. `Matrix multiplier/MatrixMultiplier.m` - the host side is a completely
   ordinary `MTL4ComputeCommandEncoder` dispatch (`setComputePipelineState`,
   `setArgumentTable`, `dispatchThreadgroups`), plus an `MTLResidencySet`
   (`newResidencySetWithDescriptor:`, `addAllocation:`, `commit`) that
   replaces the classic per-encoder `useResource:` calls - residency is
   declared once, up front, for all three tensors.
3. `Entry point/main.m` - note the problem size jumped to 512x512 (vs. 3x4
   and 4x2 in `metal_4`) and the tolerance loosened to `0.05` because the
   kernel runs in `half` precision - cooperative-tensor matmul is explicitly
   a throughput-oriented, lower-precision path, not a drop-in fp32 replacement.

---

## The Core Ideas

### 1. The command allocator is a separate, resettable object

`MTL4CommandAllocator` (`vendor/metal/metal-cpp/Metal/MTL4CommandAllocator.hpp`)
holds the memory a command buffer's encoded commands live in. A command
buffer no longer owns its own recording memory implicitly; you call
`beginCommandBuffer(allocator)` to attach one, and can `reset()` and reuse the
allocator across many command buffers instead of the queue silently managing
a buffer pool. This is the same idea as `comtam`'s own rule of "one obvious
owner per Metal object" (AGENTS.md), just applied to a piece of memory Metal
itself used to hide from you.

### 2. Argument binding is centralized into an `MTL4ArgumentTable`

```66:81:vendor/metal/metal-cpp/Metal/MTL4ArgumentTable.hpp
class ArgumentTable : public NS::Referencing<ArgumentTable>
{
public:
    MTL::Device* device() const;
    NS::String*  label() const;
    void         setAddress(MTL::GPUAddress gpuAddress, NS::UInteger bindingIndex);
    void         setAddress(MTL::GPUAddress gpuAddress, NS::UInteger stride, NS::UInteger bindingIndex);
    void         setResource(MTL::ResourceID resourceID, NS::UInteger bindingIndex);
    void         setSamplerState(MTL::ResourceID resourceID, NS::UInteger bindingIndex);
    void         setTexture(MTL::ResourceID resourceID, NS::UInteger bindingIndex);
};
```

Binding a buffer is now table mutation (`setResource` or `setAddress`), done
once, and the same table is handed to the encoder with one
`setArgumentTable(table)` call. The `setAddress(GPUAddress, ...)` overload is
the interesting one: it binds a *raw GPU virtual address*
(`MTL::Buffer::gpuAddress()`), skipping the `ResourceID` indirection entirely.
This is what people mean by "bindless" Metal - the argument table becomes a
small, cheaply-mutated array of pointers rather than a set of encoder calls
re-issued every dispatch. Compare this to `comtam`'s current per-op binding:

```53:60:comtam/core/device.cpp
    // set storage first
    encoder->setBuffer(command.a.storage->ptr(), 0, 0);
    encoder->setBuffer(command.b.storage->ptr(), 0, 1);
    encoder->setBuffer(command.out_buffer->ptr(), 0, 2);

    // then set view info
    encoder->setBytes(&command.a.view, sizeof(ViewInfo), 3);
    encoder->setBytes(&command.b.view, sizeof(ViewInfo), 4);
```

This is four `setBuffer`/`setBytes` calls issued fresh on every single op
dispatch, on a brand-new encoder, on a brand-new command buffer. An argument
table lets that binding work be paid once if the same tensors are reused
across dispatches - but only pays off once dispatch count is high enough for
per-dispatch CPU overhead to matter (see "What's Actually Relevant," below).

### 3. Pipeline compilation belongs to `MTL4Compiler`, not `MTL::Device`

Every MTL4 pipeline type (compute, render, ML, mesh, tile) is compiled
through an `MTL4Compiler` instance (`newComputePipelineState`,
`newMachineLearningPipelineState`, ...), not through `Device` directly.
`MTL4Compiler` also owns an optional `MTL4PipelineDataSetSerializer` for
saving/loading a whole set of compiled pipelines to disk. `comtam`'s
`KernelLibrary` already reaches for the same underlying idea (a cache keyed
by kernel name, `comtam/core/kernel.cpp`) but at the `MTL::Device` level,
since it never needs the render/ML/mesh pipeline types Metal 4 unifies here.

### 4. Residency is explicit and set-based, not implicit per-encoder

`metal_4_2` creates an `MTLResidencySet`, adds all three tensors to it once
(`addAllocation:` x3, then `commit`), and calls
`[commandBuffer useResidencySet:residencySet]` - one call covers every
resource the pass touches. Classic Metal's implicit residency tracking (or
manual `useResource:` per resource per encoder) is what this replaces.

### 5. `commit` on an `MTL4CommandQueue` does not block, and there is no `waitUntilCompleted()`

Both samples synchronize with an `MTLSharedEvent`: increment a signal value,
`commit`, then `signalEvent:value:` on the queue, then
`waitUntilSignaledValue:timeoutMS:` on the CPU side. `comtam`'s current
`Device::submit` (`comtam/core/device.cpp`) is unconditionally synchronous -
`command_buffer->commit(); command_buffer->waitUntilCompleted();` - which
`docs/note/METAL_USAGE.md` explicitly says is fine "while correctness is
still being established." Metal 4 does not offer that blocking call at all;
adopting MTL4 command buffers means adopting explicit event-based
synchronization at the same time, not optionally.

### 6. `MTLTensor` bakes shape into the GPU resource

A classic `MTL::Buffer` is untyped bytes; only host code (and `comtam`'s
`View`/`ViewInfo`, `comtam/core/view.h`, `comtam/core/ops.h`) knows its shape.
`MTLTensor` (`vendor/metal/metal-cpp/Metal/MTLTensor.hpp`) is a GPU resource
that carries `TensorExtents` (rank + per-dimension size) and a
`MTLTensorDataType` as part of the resource itself, with dedicated
`getBytes`/`replaceSliceOrigin` methods for strided slice copies. This is a
genuinely different tensor-metadata model from `comtam`'s "storage is dumb
bytes, `View` interprets them" split (`docs/ARCHITECTURE.md`) - worth noting
as a contrast, not a template to copy; `comtam`'s split is deliberately
simpler and the course thesis wants view/stride logic in `comtam` code you
write and test, not hidden inside a GPU resource type.

### 7. `MTL4MachineLearningCommandEncoder` runs a whole model as one pass

The `metal_4` sample's entire point is: a CoreML model (matrix
multiplication, trivially) becomes one `dispatchNetworkWithIntermediatesHeap:`
call inside your own command buffer, so its output is immediately available
to the next pass on the same timeline with no CPU round trip. This is a real
capability, but it requires a `.mtlpackage`/CoreML dependency `comtam` does
not have and should not add (AGENTS.md's "Implementation Boundaries" rules
out Python bindings and heavier dependencies generally, and CoreML is the
same category of "big adjacent framework" as ATen was for `legrad`).

### 8. MSL 4 tensor types push matmul tiling into the language and hardware

`tensor<device half, dextents<int, 2>>`, `.slice<...>`, `matmul2d_descriptor`,
and the cooperative tensor (`operation.get_destination_cooperative_tensor`,
`operation.run`, `.store`) are new Metal Shading Language 4 constructs. On
GPUs in the `apple10` family and later, `matmul2d` runs on a per-core neural
accelerator; on earlier GPUs it still runs, using SIMD-group matrix
instructions instead. Either way, the *kernel author* no longer writes the
K-loop or manages threadgroup-memory tiles - `execution_simdgroups<4>` is the
only tuning parameter exposed in this sample.

---

## What's Actually Relevant To `comtam`, And When

`comtam` is at Module 4 passed, Module 5 in progress
(`docs/course/INDEX.md`) - broadcasting, reductions, and a first matmul.
Read this section as a grounded answer to "should any of this land in
`comtam` now," not just a feature list.

**Not yet, and possibly never as a wholesale adoption:**

- Switching `Device`/`KernelLibrary`/`Command` from classic
  `MTL::CommandQueue` to `MTL4::CommandQueue` is a Module 9 hardening move at
  the earliest. It attacks per-dispatch CPU overhead (argument tables,
  reusable allocators, async commit), which is exactly what
  `docs/course/INDEX.md`'s "no optimization before a failing performance
  measurement" rule and `docs/note/AVOID.md` item 7 guard against right now.
  `comtam`'s current synchronous, per-op encoder/command-buffer path
  (`comtam/core/device.cpp`) is the correct amount of machinery for a
  framework that does not yet have reductions or matmul, let alone autograd.
- `MTLTensor` as a resource type is not worth adopting either. `comtam`'s
  `Storage` + `View` split is simpler, is the thing the course wants you to
  build and test by hand, and does not need GPU-resident shape metadata to
  work correctly at this scale.
- `MTL4MachineLearningCommandEncoder` and CoreML `.mtlpackage` models are out
  of scope entirely - wrong dependency shape for this project (see Core Idea
  7).

**Worth reading now, as orientation for Module 5's matmul assignment:**

- The `matmul2d` cooperative-tensor kernel in `metal_4_2/Shaders.metal` is
  the same kind of reference `MLX.md` calls `steel/` - "what performance work
  looks like once you stop hand-writing GEMM," not a template to imitate for
  a first matmul. A naive tiled matmul with threadgroup memory is still the
  correct Module 5 answer; this sample is useful for knowing what a
  *hardware-accelerated* answer eventually looks like, and why it needs
  `half` precision and a 64x64 tile size to make sense.
- The residency-set and argument-table patterns are useful vocabulary for
  reading Module 9 profiling notes later, even before `comtam` adopts them.

**A cheap idea worth remembering even without adopting MTL4 command buffers:**
`MTL::Buffer::gpuAddress()` and bindless-style binding exist in the *classic*
API surface too, if per-dispatch `setBuffer` overhead ever becomes the
measured bottleneck (Module 9). You do not need to adopt the entire MTL4
command model to consider that later.

---

## Concrete Files Worth Studying Closely

1. `refs/metal_example/metal_4/Matrix multiplier/MLMatrixMultiplier.m` - full
   MTL4 object lifecycle: queue, compiler, shared event, allocator, buffer.
2. `refs/metal_example/metal_4/Matrix multiplier/MLMatrixMultiplier+PipelineCompilation.m` -
   reflection-driven pipeline descriptor construction.
3. `refs/metal_example/metal_4_2/Matrix multiplier/Shaders.metal` - the
   cooperative-tensor `matmul2d` kernel; the single most important file in
   both samples for `comtam`'s matmul work specifically.
4. `refs/metal_example/metal_4_2/Matrix multiplier/MatrixMultiplier.m` -
   MTL4 compute dispatch plus `MTLResidencySet`, without the ML-encoder
   detour.
5. `vendor/metal/metal-cpp/Metal/MTL4CommandBuffer.hpp` - `beginCommandBuffer`,
   the three encoder factory methods (`computeCommandEncoder`,
   `machineLearningCommandEncoder`, `renderCommandEncoder`), `useResidencySet`.
6. `vendor/metal/metal-cpp/Metal/MTL4ArgumentTable.hpp` - `setResource` vs.
   `setAddress`, the bindless-binding surface.
7. `vendor/metal/metal-cpp/Metal/MTL4Compiler.hpp` - one compiler, many
   `new*PipelineState` methods, one per MTL4 pipeline type.
8. `vendor/metal/metal-cpp/Metal/MTLTensor.hpp` - `TensorExtents`,
   `TensorDescriptor`, `Tensor::getBytes`/`replaceSliceOrigin`.
9. `comtam/core/device.cpp` and `comtam/core/ops.h` - `comtam`'s current
   classic-Metal dispatch path, to compare against directly while reading
   the above.

---

## The Best Reading Order

### Pass 1: See what changed structurally, without the tensor pieces

Read:

- `vendor/metal/metal-cpp/Metal/MTL4CommandQueue.hpp`
- `vendor/metal/metal-cpp/Metal/MTL4CommandAllocator.hpp`
- `vendor/metal/metal-cpp/Metal/MTL4CommandBuffer.hpp`
- `vendor/metal/metal-cpp/Metal/MTL4ComputeCommandEncoder.hpp`
- `vendor/metal/metal-cpp/Metal/MTL4ArgumentTable.hpp`

Goal: understand allocator/buffer/encoder/argument-table as four separate,
reusable objects, purely as a command-submission API change, before tensors
enter the picture at all.

### Pass 2: Read the compute sample end to end

Read `refs/metal_example/metal_4_2/` in this order: `Matrix/Matrix.h`,
`Matrix multiplier/MatrixMultiplier.h`, `Matrix multiplier/MatrixMultiplier.m`,
`Matrix multiplier/Shaders.metal`, `Entry point/main.m`.

Goal: trace one matmul from host-side tensor creation through
`dispatchThreadgroups` into the cooperative-tensor kernel and back.

### Pass 3: Read the machine-learning-pass sample end to end

Read `refs/metal_example/metal_4/` in this order: `Entry point/main.m`,
`Matrix multiplier/MLMatrixMultiplier.h`,
`Matrix multiplier/MLMatrixMultiplier.m`,
`Matrix multiplier/MLMatrixMultiplier+PipelineCompilation.m`,
`Matrix multiplier/MLMatrixMultiplier+TensorSetup.m`.

Goal: see the same MTL4 command/allocator/argument-table plumbing reused for
a completely different encoder type (`MachineLearningCommandEncoder`
instead of `ComputeCommandEncoder`), confirming that the command API and the
ML/tensor features are independent axes.

### Pass 4: Compare against `comtam` directly

Read:

- `docs/ARCHITECTURE.md` (`Device`, `KernelLibrary`, `Operator Dispatch`
  sections)
- `comtam/core/device.cpp`
- `comtam/core/ops.h`
- `docs/note/METAL_USAGE.md`

Goal: list, concretely, which lines of `Device::submit` would have to change
if `comtam` moved to `MTL4CommandQueue` today, and decide (per "What's
Actually Relevant" above) that the answer is "not yet."

---

## Questions To Ask While Reading

1. Why does Metal 4 split "recording memory" (allocator) from "command
   buffer" when classic Metal folded them together? What CPU cost does that
   split let an app amortize that classic Metal could not?
2. Why is pipeline compilation an `MTL4Compiler` method instead of a
   `MTL::Device` method in Metal 4, when it was a `Device` method in classic
   Metal?
3. Why does Metal 4 remove `waitUntilCompleted()` entirely instead of just
   adding an async alternative alongside it? What does that force every MTL4
   adopter to confront up front?
4. Why does the `matmul2d` sample use `half` precision and a fixed 64x64
   tile, and what would change about the kernel signature if `comtam` needed
   this at fp32 (the course's stated float32-first rule)?
5. `MTL4ArgumentTable::setAddress` binds a raw `MTL::GPUAddress` instead of a
   `MTL::ResourceID`. What has to be true about buffer lifetime for that to
   be safe, and how does that compare to `NS::SharedPtr<MTL::Buffer>`
   ownership in `comtam::core::Storage`?

---

## Final Takeaway

Metal 4 is not one feature `comtam` should "upgrade to." It is at least four
separable ideas bundled under one marketing name:

- a lower-overhead command-submission API (allocator, argument table,
  compiler, explicit residency, async commit),
- a GPU-resident tensor resource type,
- an inline machine-learning encoder for running whole CoreML models on the
  GPU timeline, and
- new MSL 4 language-level tensor/cooperative-matmul constructs.

For `comtam` right now, only one of these is worth reading closely: the
cooperative-tensor `matmul2d` kernel, purely as orientation for what a
"three years later, hardware-accelerated" matmul looks like while Module 5
is still asking for a correct, hand-written, threadgroup-tiled one. The
command-submission API is real and its performance argument is real, but it
is Module 9 territory - it optimizes per-dispatch CPU overhead in a framework
that does not have enough dispatches yet to measure that overhead
meaningfully. `MTLTensor` and the ML command encoder are not worth adopting
at all at `comtam`'s current scope. Read Metal 4 to know what exists and why
it exists, not to import it early.
