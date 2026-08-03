## Appendix A: Current comtam Code Map

This table describes current source, not files promised by later modules.

| File | Purpose | Introduced by |
| --- | --- | --- |
| `comtam/core/context.h/.cpp` | explicit owner of device and kernel library | Module 3 |
| `comtam/core/device.h/.cpp` | Metal device/queue, allocation, copies, submission | Modules 1 and 3 |
| `comtam/core/storage.h/.cpp` | one owner of one `MTL::Buffer` | Module 1 |
| `comtam/core/command.h` | GPU-facing command and view descriptors | Modules 3 and 5 |
| `comtam/core/kernel.h/.cpp` | precompiled library loading and pipeline cache | Module 3 |
| `comtam/tensor/tensor.h/.cpp` | tensor metadata, host transfer, public ops | Modules 1-5 |
| `comtam/tensor/view.h/.cpp` | shape, strides, offset, and movement operations | Module 2 |
| `comtam/tensor/dtype.h` | dtype dispatch, float32 first | Module 1 |
| `comtam/tensor/op.h` | operation identifiers | Modules 3 and 5 |
| `comtam/kernels/*.metal` | Metal implementations | Modules 3 and 5 |
| `comtam/utils/rng.h` | deterministic input/initialization support | Modules 4 and 7 |
| `tests/support/` | forward oracles and shared assertions | Module 4 onward |

Update this map after each passed module. Planned files for autograd, `nn`,
persistence, or Python do not belong here until they exist.

Supporting project docs:

- [`../ARCHITECTURE.md`](../../../ARCHITECTURE.md) - architecture that currently exists
- [`../note/METAL_USAGE.md`](../../../note/METAL_USAGE.md) - Metal ownership and errors
- [`../note/AVOID.md`](../../../note/AVOID.md) - premature abstractions to avoid
- [`../note/MLX_C_ORACLE.md`](../../../note/MLX_C_ORACLE.md) - test-only MLX C usage

## Appendix B: Reference Frameworks

Use references after your own attempt, when you have a concrete question.
Study guides for each live in [`../refs/`](../../../refs/).

- `refs/magnetron` ([`../refs/MAGENETRON.md`](../../../refs/MAGENETRON.md)) is the
  closest eager-runtime reference for tensor identity, autograd, state
  dictionaries/snapshots, Python bindings, packaging, and the cost of a much
  broader backend/dtype/operator scope.
- `refs/mlx` ([`../refs/MLX.md`](../../../refs/MLX.md), with
  [`MLX_KERNELS.md`](../../../refs/MLX_KERNELS.md) and
  [`MLX_REDUCTION.md`](../../../refs/MLX_REDUCTION.md)) is the production Apple GPU
  kernel reference: kernel organization, naming, and reduction design.
- `refs/luminal` ([`../refs/LUMINAL.md`](../../../refs/LUMINAL.md), with
  [`LUMINAL_METAL.md`](../../../refs/LUMINAL_METAL.md)) is the compiler-first
  counterpoint. Read it for the RISC-style primitive op set behind Module 5A's
  semantic-surface audit and for its production Metal backend, not as an architecture
  template - its lazy graph, e-graph search, and symbolic shapes are exactly
  what comtam's eager thesis excludes.

Suggested reading order:

1. Trace Magnetron storage/tensor/view ownership after Modules 1-2.
2. Trace one API-to-kernel dispatch after Module 3.
3. Read Luminal's primitive op set and frontend compositions while working
   through Module 5A; read `LUMINAL_METAL.md` when a Metal question
   outgrows comtam's own backend.
4. Read dynamic autograd only after attempting Module 6.
5. Read state/snapshot code only after writing Module 12's format contract.
6. Read bindings and Python package structure only after defining Module 14's
   ownership model.
7. Read backend registries, allocator machinery, and broad dtype dispatch only
   when a measured comtam requirement makes the comparison relevant.

Borrow the reason for an abstraction, not the amount of code surrounding it.

## Appendix C: metal-cpp Debugging Tips

These are diagnostic branches, not substitutes for the module's required tests.

**`failed to create default Metal device`**

- Confirm a Metal device exists and whether the process is sandboxed/headless.
- Record the environment failure separately from a code failure.
- Run outside a sandbox only with explicit permission.

**A kernel produces wrong values**

- Compare buffer indices and byte sizes with the Metal function signature.
- Check grid bounds and the zero-element no-dispatch path.
- Check input shape, stride, offset, and rank metadata independently.
- Compare the smallest failing shape to the CPU/MLX oracle.

**A Metal library or pipeline fails**

- Preserve the complete `NS::Error` message.
- Confirm the build produced and packaged `default.metallib`.
- Confirm the op/dtype name maps to an exported kernel function exactly.

**Training memory grows**

- Inspect deterministic live tensor/autograd/in-flight counts first.
- Confirm per-step graphs are consumed and gradients are cleared intentionally.
- Confirm autorelease pools and async completion retirement run at their defined
  boundaries.
- Treat RSS as supporting evidence because driver/allocator caches retain memory.

**Async work appears to succeed silently**

- Synchronize at a documented boundary and inspect command-buffer status.
- Retain operation context until deferred errors are reported.
- Confirm resources remain owned until GPU completion.

## Appendix D: Index Decomposition Cheat Sheet

For logical shape `(d0, d1, d2)` and flat `linear_idx`:

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

Movement rules:

```text
transpose/permute : reorder shape and strides, preserve offset
shrink             : offset += sum(start_i * stride_i), preserve strides
expand             : introduced/expanded size-one axes receive stride 0
reshape            : change shape/strides only when layout remains representable
contiguous         : allocate and gather logical values (Module 10)
```

Autograd inverts a user-authored `shrink` by materializing zero-padding through
Module 6's internal `zero_pad` path. That backward-support kernel is not a
public movement operation and does not change the forward view rule above.

## Appendix E: Gate Evidence Template

Use this block in each `docs/archive/course-v1/solution/MODULE_N.md` grading section:

```text
configured     -> exact CMake/Python configuration
compiled       -> exact build command and result
unit tested    -> exact targeted test command and result
integrated     -> training/stress/consumer command and result
gate passed    -> each numbered exit criterion mapped to evidence
not run        -> environmental or scope gaps stated explicitly
```

Never collapse these into "works." A build does not prove execution, and a smoke
run does not prove a module gate.
