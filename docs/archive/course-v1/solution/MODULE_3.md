## Assignment 3.1: Trace The Submit Path

1. How does it get a pipeline from the Op? Trace KernelLibrary::get (kernel.cpp) and the name lookup in op_to_kernel_name (ops.h:17).

**Answer**

The kernel library will lazily compute pipeline for each op. If op isn't supported => throw error. If op is support, we have two case:
- Already cached in `pipeline_cache_` => return.
- Else => create `function` for requsted op. Then, create pipeline by `device_->newComputePipelineState` with the corresponding `function`.

We lookup for op name in `op_to_kernel_name` (a hashmap from `Op` to string) but we don't take account of `DType`.

2. What is the exact encoder sequence: create command buffer, create compute encoder, set pipeline, bind buffers at indices, set threadgroup sizes, endEncoding, commit, waitUntilCompleted?

**Answer**

The full sequence is:
```text
(1) get pipeline for op
-> (2) create new command buffer 
-> (3) create new command encoder 
-> (4) encode the pipeline to command encoder
-> (5) set threadgroup sizes
-> (6) dispatch threads and end encoding
-> (7) commit command to GPU for execution
-> (8) wait until completed and check for error
```

3. At which buffer indices are a, b, and out bound, and do those indices match the argument order in binary.metal?

**Answer**

In step (6), we need to set Buffer to its correct index, for example, we have a metal kernel as below:

```metal
kernel void add(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] + inB[index];
}
// Do result = inA + inB
```

Then `c = a + b` tensor will map to `a` as index `0`, `b` as index `(0, 1)` and `c` as index `(0, 2)` (same index as input argument).

4. How is the threadgroup size chosen, and what happens when elements is not a multiple of it? Does the kernel guard against out-of-range thread ids?

**Answer**

First, we get the number of threads per group by using `pipeline->threadExecutionWidth`, this funciton will return the number of threads that the GPU executes simultaneously [^1]. Note that we cank think this as `warpSize` in `CUDA`.

Then we set number of threads in grid to maximum number of elements (`command.elements`), this maybe work for element-wise kernel.

What about when elements is not a multiple of it? Metal can generate smaller threadgroups along the edges of your grid. Compared to uniform threadgroups, this technique simplifies kernel code and improves GPU performance [^2].

[^1]: https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/threadexecutionwidth
[^2]: https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes

### Agent Feedback / Grading

Status: mostly passed for Assignment 3.1, with one important correction.

What is good:

- You traced the intended eager path correctly: pipeline lookup, command buffer,
  compute encoder, pipeline binding, buffer binding, dispatch, end encoding,
  commit, wait, then error check.
- You correctly noticed that the original op lookup only considered `Op`, while
  a real kernel key should also include `DType`.
- The `threadExecutionWidth()` explanation is the right local mental model for
  choosing a simple first threadgroup width.

Corrections:

- The buffer index answer has a typo/confusion. The binding should be `a -> 0`,
  `b -> 1`, and `out -> 2`. Do not write `b` as `(0, 1)` or `c` as `(0, 2)`.
- The step list skips the explicit buffer bindings. They are part of the
  contract, not a detail hidden inside "dispatch threads".
- Your answer talks about out-of-range protection, but the current templated
  kernels do not take `elements` and do not guard `id >= elements`.

## Assignment 3.2: Confirm The Kernel Contract

1. Yes
2. No, it could go wrong if we didnt have `if (gid >= elements)`, the kernel would be compute garbage value (out of bound access).
3. As we said in Assignment 3.1, the dispatch should take account of `DType` too.

Here is new kernel that supports multiple type:
```metal
#include <metal_stdlib>

using namespace metal;

/* --- Add Kernel --- */
// https://stackoverflow.com/questions/56687496/how-to-make-templated-compute-kernels-in-metal
template<typename T>
kernel void add(
        device const T* src0,
        device const T* src1,
        device T* dst,
        uint id [[thread_position_in_grid]]) {
    dst[id] = src0[id] + src1[id];
}

template [[ host_name("add_fp32") ]]
kernel void add(device const float*, device const float*, device float*, uint);
```

### Agent Feedback / Grading

Status: passed after the kernel contract update.

The written checklist is too thin for this assignment. "Yes / No" does not give
enough evidence that the kernel contract was actually checked against the host
dispatch code.

What is good:

- Moving toward dtype-qualified names such as `add_fp32` is a reasonable design
  for the next small step.
- The templated Metal source avoids duplicating the arithmetic body for every
  dtype. That is fine while only one dtype is instantiated.

Previous blocking issues now resolved:

- The assignment asks you to confirm the actual names, bounds guard, and buffer
  types. The answer should explicitly say what names exist in
  `binary.metal`, what names `KernelLibrary::get` asks for, and whether those
  match exactly.
- The current kernels now take `N` as a constant argument and guard `id >= N`.
- `Device::submit` binds `command.elements` at buffer/bytes index 3 with the
  correct byte length.

## Assignment 3.3: Implement The Binary Op Layer

```cpp
// ----- Binary operation -----
Tensor Tensor::binop(const Tensor &a, const Tensor &b, const core::Op& op, core::Context& ctx) {
    auto& device = ctx.device();
    auto& kernels = ctx.kernels();

    if (a.dtype_ != b.dtype_) {
        throw std::runtime_error("Two operands must have the same dtype");
    }

    if (!a.view_.is_contiguous() || !b.view_.is_contiguous()) {
        throw std::runtime_error("Cannot do operation on non-contiguous inputs");
    }

    if (a.view_ != b.view_) {
        throw std::runtime_error("Two operands must have the same shape");
    }

    // create an empty tensor
    Tensor out(a.view_, device, a.dtype_);

    // first we create a command
    core::Command cmd = {
        .kernel = {.op = op, .dtype = a.dtype_},
        .a = a.storage_.get(),
        .b = b.storage_.get(),
        .out = out.storage_.get(),
        .elements = a.numel(),
    };

    // then we submit the command
    device.submit(cmd, kernels);

    return out;
}

Tensor Tensor::add(const Tensor &a, const Tensor &b, core::Context& ctx) {
    return Tensor::binop(a, b, core::Op::ADD, ctx);
}

Tensor Tensor::sub(const Tensor &a, const Tensor &b, core::Context& ctx) {
    return Tensor::binop(a, b, core::Op::SUB, ctx);
}

Tensor Tensor::mul(const Tensor &a, const Tensor &b, core::Context& ctx) {
    return Tensor::binop(a, b, core::Op::MUL, ctx);
}

Tensor Tensor::div(const Tensor &a, const Tensor &b, core::Context& ctx) {
    return Tensor::binop(a, b, core::Op::DIV, ctx);
}
```

### Agent Feedback / Grading

Status: passed after adding the missing tests.

What is good:

- The op layer follows the eager design: validate, allocate output, build a
  `Command`, submit through `Device::submit`, return the result.
- The Tensor op layer does not touch Metal directly.
- DType, contiguity, and exact-view checks are in the right place: Tensor policy
  lives above the flat command dispatch layer.
- `add`, `sub`, `mul`, and `div` all share the same `binop` path, which is the
  right amount of abstraction here.

Tests added during review:

- External-oracle checks for `add/sub/mul/div` on asymmetric shapes `(2, 3)`,
  `(3, 4)`, and `(2, 3, 4)`.
- Throw tests for mismatched shapes.
- Throw tests for non-contiguous inputs.

Remaining caution:

- The current equality check compares the full `View`, not just shape. That is
  acceptable for this module because the op requires contiguous same-shape
  inputs. If later modules allow non-contiguous or broadcasted inputs, this
  check will need to become shape/broadcast validation instead of full-view
  equality.

## Assignment 3.4: Reduce Op Enum Duplication

```cpp
inline std::string op2kernel(const Op& op) {
    switch (op) {
        case Op::ADD:
            return "add";
        case Op::SUB:
            return "sub";
        case Op::MUL:
            return "mul";
        case Op::DIV:
            return "div";
        default:
            throw std::runtime_error("op isn't supported by kernel");
    }
}

inline std::string dtype2kernel(const DType& dtype) {
    switch (dtype) {
        case DType::Float32:
            return "fp32";
        default:
            throw std::runtime_error("dtype isn't supported by kernel");
    }
}

MTL::ComputePipelineState *KernelLibrary::get(const Kernel &kernel) {
    std::string name = op2kernel(kernel.op) + "_" + dtype2kernel(kernel.dtype);
    ...
}
```

### Agent Feedback / Grading

Status: passed after enabling missing-enum diagnostics.

What is good:

- Replacing the header-level `static std::unordered_map` with a small `switch`
  is the right direction. Four ops do not need hash-map machinery.
- Combining `op2kernel(op)` and `dtype2kernel(dtype)` inside
  `KernelLibrary::get` makes the current kernel naming policy easy to inspect.

Compile-time coverage:

- The `default:` branch was removed from `op2kernel`.
- `comtam_lib` now builds with `-Wswitch-enum -Werror=switch-enum` on Clang-like
  compilers, so adding a new `Op` without updating the switch fails the build.

Small cleanup:

- `ops.h` includes `<mach/mach_traps.h>`, but nothing in that header appears to
  use it. Drop unrelated includes unless a compiler error proves they are
  needed.

## Assignment 3.5: One Autorelease Pool Per Launch Boundary

Right now, we create a new pool in each `Device::submit` with a tradeoff: prompt draining after each blocking submit, at the cost of per-launch pool overhead.

### Agent Feedback / Grading

Status: passed as a written policy, assuming the code intentionally keeps the
pool inside `Device::submit`.

This is a reasonable policy for the current synchronous runtime:

- `Device::submit` calls `waitUntilCompleted()`, so each launch boundary is also
  a natural lifetime boundary for autoreleased command-buffer/encoder objects.
- A pool per submit drains promptly and keeps future loops from accumulating
  Metal temporaries.
- The cost is per-launch pool overhead, which is acceptable until a failing
  performance measurement says otherwise.

One caveat: if `Device::submit` becomes asynchronous later, this policy must be
revisited because the launch boundary and completion boundary will no longer be
the same thing.

## Module 3 Verdict

Status: passed.

Verification:

```text
compiled        -> passed with `cmake --build build`
smoke ran       -> passed with `ctest --test-dir build --output-on-failure`
gate passed     -> passed
```

Commands run:

```text
cmake --build build
ctest --test-dir build --output-on-failure
```

CTest passed with 20/20 tests after adding the asymmetric-shape and throw tests
and after updating the kernel bounds contract.

What is now solid:

- The eager binary op path exists for `add/sub/mul/div`.
- Existing binary ops match MLX C oracles on asymmetric shapes.
- Mismatched shapes and non-contiguous inputs are rejected by tests.
- Every public Tensor binary op goes through `Device::submit`.

Why the gate now passes:

- `add(a, b, ctx)` and the other binary ops run on the GPU and match CPU
  oracles on asymmetric shapes.
- Every public Tensor binary op routes through `Device::submit`.
- Buffer indices, kernel names, dtype-qualified kernel lookup, and the bounds
  guard agree.
- Adding a new `Op` without updating `op2kernel` now fails the build under the
  configured switch-enum warning policy.
- The autorelease pool policy is written down and matches the current
  synchronous launch boundary.
