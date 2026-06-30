## Appendix A: comtam Code Map

The current source, with the module that owns each concept.

| File | Purpose | Owned By Module |
| --- | --- | --- |
| `comtam/core/context.h/.cpp` | top-level owner of Device and KernelLibrary | Module 3 dispatch |
| `comtam/core/device.h/.cpp` | MTL::Device, command queue, allocate, copy, submit | Modules 1 and 3 |
| `comtam/core/storage.h/.cpp` | one owner of one MTL::Buffer | Module 1 |
| `comtam/core/view.h` | shape, strides, ref_strides, offset, contiguity | Module 2 |
| `comtam/core/dtype.h` | DType and byte sizes (float32 first) | Module 1 |
| `comtam/core/ops.h` | Op enum, kernel-name lookup, Command struct | Module 3 |
| `comtam/core/kernel.h/.cpp` | KernelLibrary, pipeline cache | Module 3 |
| `comtam/kernels/binary.metal` | elementwise Metal kernels | Module 3 |
| `comtam/tensor.h/.cpp` | Tensor: View + Storage + DType, host transfer | Modules 1, 2 |
| `comtam/utils/rng.h` | random data generation for tests and init | Modules 4, 7 |
| `comtam/main.cpp` | current entry point and smoke check | Project shell |

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

## Appendix D The Index Decomposition Cheat Sheet

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
