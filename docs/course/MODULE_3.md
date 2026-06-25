## Module 3: Eager Operator Dispatch

Make one real op run on the GPU, end to end, with one obvious owner for every
Metal object.

## Why This Module Exists

This is where comtam stops being a memory manager and becomes a tensor
framework.

The whole eager design lives in one short path:

```text
add(a, b)
  validate shapes and dtypes
  allocate result storage
  build a Command
  Device::submit(command, kernels)
  return result Tensor
```

No lazy graph. No scheduler. No fusion. The op runs now. The discipline that
matters is the Magnetron-inspired uniform path from
[`../ARCHITECTURE.md`](../ARCHITECTURE.md):

```text
high-level op -> uniform Command -> Device::submit
```

Do not hide kernel launches inside unrelated helpers. Every launch goes through
`Device::submit`.

## What Already Exists

comtam already has the skeleton:

```text
comtam/core/ops.h        Op enum (ADD, SUB, MUL, DIV), op_to_kernel_name, Command
comtam/core/kernel.h     KernelLibrary::get(Op) -> MTL::ComputePipelineState*
comtam/core/device.h     Device::submit(const Command&, KernelLibrary&)
comtam/kernels/binary.metal   the elementwise kernels
```

`Command` (`ops.h:30`) is intentionally binary-only for now:

```cpp
struct Command {
    Op op;
    Storage* a;
    Storage* b;
    Storage* out;
    size_t elements;
};
```

This module wires a public `Tensor` op on top of that path, with tests.

## Mental Model

Three responsibilities, kept separate:

```text
Tensor op layer   shape/dtype checks, result allocation, autograd metadata later
Command           a flat description of one kernel launch
Device::submit    pipeline lookup, encoder, buffer binding, dispatch, sync
```

The op layer never touches Metal. `Device::submit` never makes policy decisions
about shapes. The `Command` is the contract between them.

## Assignment 3.1: Trace The Submit Path ⭐⭐

**Task:** Before writing the op layer, fully trace `Device::submit` in
`device.cpp`.

Questions:

1. How does it get a pipeline from the `Op`? Trace `KernelLibrary::get`
   (`kernel.cpp`) and the name lookup in `op_to_kernel_name` (`ops.h:17`).
2. What is the exact encoder sequence: create command buffer, create compute
   encoder, set pipeline, bind buffers at indices, set threadgroup sizes,
   `endEncoding`, `commit`, `waitUntilCompleted`?
3. At which buffer indices are `a`, `b`, and `out` bound, and do those indices
   match the argument order in `binary.metal`?
4. How is the threadgroup size chosen, and what happens when `elements` is not a
   multiple of it? Does the kernel guard against out-of-range thread ids?

**Deliverable:** A labeled diagram of the submit path, plus a one-line note on
whether buffer indices and kernel signatures agree.

**Why this assignment:** A wrong buffer index or a missing bounds guard produces
garbage that looks like a math bug. Trace it before you trust it.

## Assignment 3.2: Confirm The Kernel Contract ⭐

**Task:** Read `comtam/kernels/binary.metal` and confirm it matches the dispatch
assumptions.

Questions:

1. Do the kernel function names match `op_to_kernel_name` exactly (`add`, `sub`,
   `mul`, `div`)?
2. Does each kernel guard `if (gid >= elements) return;`? What goes wrong on the
   last threadgroup if it does not?
3. What is the type of the buffers (`device float*`)? What happens if the host
   binds a buffer sized for a different element count?

**Deliverable:** A checklist confirming names, bounds guard, and types all agree.
File any mismatch in [`BUG_LEDGER.md`](BUG_LEDGER.md).

## Assignment 3.3: Implement The Binary Op Layer ⭐⭐

**Task:** Add a public elementwise op that produces a new `Tensor`.

```cpp
Tensor add(const Tensor& a, const Tensor& b, Context& ctx);
// and sub, mul, div
```

The op should:

1. Validate that shapes match exactly (broadcasting comes in Module 5) and that
   dtypes match.
2. Allocate a result `Tensor` with the same shape, contiguous.
3. Build a `Command` with the three storages and `elements = numel()`.
4. Call `ctx.device().submit(command, ctx.kernels())`.
5. Return the result.

Rules:

- The op layer must not call Metal directly. Only through `Device::submit`.
- For now require contiguous, same-shape inputs. Reject anything else with a
  clear error. Honest restrictions beat silent wrong answers.

**Tests to write (against a CPU oracle):**

```text
add: c[i] == a[i] + b[i]   for asymmetric shapes like (2,3), (3,4), (2,3,4)
sub, mul, div: same pattern
mismatched shapes: throws
non-contiguous input: throws (until Module 5)
```

**Why this assignment:** This is the first op that runs on the GPU and is checked
against a reference. It sets the template every later op will follow.

## Assignment 3.4: Reduce Op Enum Duplication ⭐

**Observation:** `op_to_kernel_name` (`ops.h:17`) is a `static` map in a header.
Included in multiple translation units, that creates one copy per TU.

**Task:** Decide on one home for the op-to-name mapping.

Questions:

1. Should the map be `inline`, or moved into a `.cpp`?
2. Is a `switch` over the `Op` enum clearer and cheaper than a hash map for four
   ops?
3. Does the `enum class Op` need an exhaustive `switch` so adding an op forces a
   compile error at every dispatch site?

**Recommended first choice:** Replace the map with a small `switch` that returns
the kernel name, and make the default case a hard error. With four ops, a map is
more machinery than the problem needs (see [`../AVOID.md`](../AVOID.md)). An
exhaustive switch turns "forgot to handle the new op" into a build failure.

## Assignment 3.5: One Autorelease Pool Per Launch Boundary ⭐⭐

**Task:** Confirm Metal temporaries do not leak across many op calls.

Background: command buffers and encoders are autoreleased. Without a pool, a
training loop that submits thousands of commands can accumulate them. See
[`../NOTE.md`](../NOTE.md) on autorelease pools.

Questions:

1. Where should the autorelease pool live: in `main`, in `submit`, or in the
   training loop?
2. If `submit` blocks on `waitUntilCompleted`, does each call's pool drain
   promptly?

**Recommended first choice:** Keep a pool in `main` now, and plan a per-iteration
pool for the Module 8 training loop. Do not over-engineer pooling before a loop
exists.

**Why this assignment:** Leaks are invisible until a long run, then they look
like a different bug entirely. Decide the policy before Module 8.

## Module 3 Checklist

- [ ] 3.1 Trace `Device::submit` end to end.
- [ ] 3.2 Confirm `binary.metal` matches the dispatch contract.
- [ ] 3.3 Implement `add/sub/mul/div` op layer with CPU-oracle tests.
- [ ] 3.4 Collapse op-name lookup into an exhaustive switch.
- [ ] 3.5 Decide the autorelease pool policy.

## Exit Criteria

You are ready for Module 4 when:

1. `add(a, b, ctx)` runs on the GPU and matches a CPU oracle on asymmetric
   shapes.
2. Every kernel launch goes through `Device::submit`; no op touches Metal
   directly.
3. Buffer indices, kernel names, and the bounds guard all agree, verified.
4. Adding a new `Op` causes a compile error until every switch handles it.
5. You have a written autorelease pool policy for the eventual training loop.
