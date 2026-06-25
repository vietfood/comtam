## Appendix A: comtam Code Map

The current source, with the module that owns each concept.

| File | Purpose | Owned By Module |
| --- | --- | --- |
| `comtam/core/context.h/.cpp` | top-level owner of Device and KernelLibrary | Module 0 |
| `comtam/core/device.h/.cpp` | MTL::Device, command queue, allocate, copy, submit | Modules 0, 3 |
| `comtam/core/storage.h/.cpp` | one owner of one MTL::Buffer | Module 1 |
| `comtam/core/view.h` | shape, strides, ref_strides, offset, contiguity | Module 2 |
| `comtam/core/dtype.h` | DType and byte sizes (float32 first) | Module 1 |
| `comtam/core/ops.h` | Op enum, kernel-name lookup, Command struct | Module 3 |
| `comtam/core/kernel.h/.cpp` | KernelLibrary, pipeline cache | Modules 0, 3 |
| `comtam/kernels/binary.metal` | elementwise Metal kernels | Module 3 |
| `comtam/tensor.h/.cpp` | Tensor: View + Storage + DType, host transfer | Modules 1, 2 |
| `comtam/utils/rng.h` | random data generation for tests and init | Modules 4, 7 |
| `comtam/main.cpp` | current entry point and smoke check | Module 0 |

Supporting project docs:

- [`../ARCHITECTURE.md`](../ARCHITECTURE.md) - intended layering and ownership
- [`../NOTE.md`](../NOTE.md) - metal-cpp ownership, errors, autorelease pools
- [`../AVOID.md`](../AVOID.md) - what comtam must not adopt early, and why

## Appendix B: Reference Frameworks

Use these as contrast, after your own attempt. Borrow the arc, not the size.

- `refs/magnetron` - architecture reference for tensor metadata, view ownership,
  operator dispatch, and dynamic autograd. The closest match to comtam's
  eager/runtime shape.
- `refs/banhxeo` - a compiler-first, lazy, Python framework. Useful contrast for
  scheduling and codegen, but it pulls toward graph internals comtam is
  deliberately avoiding for now.
- `refs/legrad` - anti-pattern reference. Early allocator and backend complexity
  is exactly what [`../AVOID.md`](../AVOID.md) warns against.

Suggested study order:

1. Read `refs/magnetron` AGENTS/architecture for the high-level shape.
2. Trace one eager op through Magnetron from API to dispatch.
3. Read Magnetron storage and view ownership (Modules 1 and 2).
4. Read Magnetron autograd only after attempting Module 6 yourself.

Why this order: do not start with the most advanced abstraction. Start with
ownership and dispatch so you know what pressure your own code is under before
you import a solution.

## Appendix C: metal-cpp Debugging Tips

From [`../NOTE.md`](../NOTE.md), the failures you will actually hit:

**"failed to create default Metal device"**

- No Metal device available, often a sandbox or headless environment.
- State it explicitly; run outside the sandbox only with permission.

**Kernel produces wrong values**

- Check buffer bind indices against the kernel argument order.
- Check the bounds guard `if (gid >= elements) return;` on the last threadgroup.
- Verify the view indexing math for non-contiguous inputs (Module 2).

**Kernel fails to compile / library load fails**

- Convert `NS::Error*` to a readable message; do not swallow it.
- Confirm kernel function names match the dispatch name lookup (`ops.h`).

**Memory grows during training**

- Add a per-iteration autorelease pool (Modules 3 and 8).
- Confirm the autograd graph from each step is released.
- Confirm `zero_grad` runs each step.

**Command buffer silently does nothing**

- Check `command_buffer->status()` after `waitUntilCompleted` and throw on
  `MTL::CommandBufferStatusError`.

## Appendix D: Checklist By Module

### Module 0: Current Codebase
- [ ] 0.1 Trace tensor construction
- [ ] 0.2 Trace the round trip
- [ ] 0.3 Trace the Command dispatch path
- [ ] 0.4 Build and smoke run
- [ ] 0.5 Start the bug ledger

### Module 1: Storage And Metadata
- [ ] 1.1 Trace buffer lifetime
- [ ] 1.2 Make element/byte conversion explicit
- [ ] 1.3 Prove storage sharing is safe
- [ ] 1.4 Storage debug print
- [ ] 1.5 Read Magnetron storage

### Module 2: Views And Strides
- [ ] 2.1 Verify contiguous stride construction
- [ ] 2.2 Implement physical_offset
- [ ] 2.3 Implement permute/transpose
- [ ] 2.4 Implement shrink/slice
- [ ] 2.5 Implement expand
- [ ] 2.6 Implement reshape with clear errors
- [ ] 2.7 Make to_vector view-aware
- [ ] 2.8 Movement gradient preview
- [ ] 2.9 Read mature view code

### Module 3: Eager Dispatch
- [ ] 3.1 Trace submit path
- [ ] 3.2 Confirm kernel contract
- [ ] 3.3 Implement binary op layer
- [ ] 3.4 Collapse op-name lookup to a switch
- [ ] 3.5 Autorelease pool policy

### Module 4: Forward Correctness
- [ ] 4.1 Forward comparison helper
- [ ] 4.2 Test elementwise ops
- [ ] 4.3 Test movement ops via readback
- [ ] 4.4 Write the correctness map

### Module 5: Broadcast/Reduce/Matmul
- [ ] 5.1 broadcast_shapes
- [ ] 5.2 Broadcasting binary ops
- [ ] 5.3 Full reduce sum
- [ ] 5.4 Axis sum/mean
- [ ] 5.5 Naive matmul

### Module 6: Autograd
- [ ] 6.1 Design the grad node
- [ ] 6.2 Backward elementwise
- [ ] 6.3 Backward movement ops
- [ ] 6.4 Broadcasting grad accumulation
- [ ] 6.5 Backward reduce/matmul
- [ ] 6.6 Read Magnetron autograd

### Module 7: nn And SGD
- [ ] 7.1 Parameter and parameters()
- [ ] 7.2 Linear
- [ ] 7.3 relu and MSE
- [ ] 7.4 SGD with zero_grad
- [ ] 7.5 Smallest training loop

### Module 8: Training
- [ ] 8.1 Load MNIST
- [ ] 8.2 Stable cross-entropy
- [ ] 8.3 Train the MLP
- [ ] 8.4 Audit leaks and graph lifetime

### Module 9: Hardening
- [ ] 9.1 Benchmark harness
- [ ] 9.2 Find the real bottleneck
- [ ] 9.3 Optimize the top one only
- [ ] 9.4 Harden the failure surface
- [ ] 9.5 Decide what to earn next

## Appendix E: The Index Decomposition Cheat Sheet

For logical shape `(d0, d1, d2)` and a flat `linear_idx`:

```text
i2 = linear_idx % d2
t  = linear_idx / d2
i1 = t % d1
t  = t / d1
i0 = t % d0

physical_offset = offset + i0*stride0 + i1*stride1 + i2*stride2
```

Contiguous fast path:

```text
physical_offset = linear_idx + offset
```

Movement op stride rules:

```text
transpose/permute : reorder strides, offset unchanged
slice/shrink      : offset += sum(start_i * stride_i), strides unchanged
expand            : size-1 axis gets stride 0
flip              : stride negated, offset += (size-1)*stride
```
