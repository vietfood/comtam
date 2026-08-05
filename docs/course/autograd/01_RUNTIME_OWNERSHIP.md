# Chapter 1: Runtime Ownership Before Autograd

**Status:** Implemented; completion evidence pending.

[Autograd track index](INDEX.md) | [Decision reference](DECISIONS.md)

This chapter does not assume that names such as `runtime_state`, `tensor_impl`, `grad_fn`, or "tape" already mean anything to you. It starts from a concrete problem in the current source, considers the simplest designs, explains the tradeoff we are choosing, and turns the choice into small implementation steps and tests.

Do not treat the code sketches as a patch to paste blindly. Type them in deliberately, compare them with the current classes, and stop whenever you cannot explain who owns an object and how long it lives. Ownership is the difficult part of C++ autograd; derivative formulas are usually the easier part.

## Where We Are Starting

Today, `core::context` directly owns one `metal_device` and one `kernel_library`:

```text
core::context
├── unique_ptr<metal_device>
└── unique_ptr<kernel_library>
```

Today, `tensor` is a value containing dtype and view metadata plus shared Metal storage:

```text
tensor
├── DType
├── view
└── shared_ptr<storage>
```

Every arithmetic operation receives an explicit context:

```cpp
auto y = tensor::mul(a, b, ctx);
auto z = tensor::sum(y, ctx);
```

This design has served the forward runtime well because ownership is obvious: the caller creates the context and passes it to every operation. Autograd changes the pressure on this API. A produced tensor must remember enough runtime identity to execute its backward rules later, and we want eventual C++ and Python code to say `loss.backward()` without passing the same context through every expression.

The first problem is therefore not differentiation. The first problem is answering this question:

> If an operation no longer receives `ctx`, where do its device and kernels come from, and who keeps them alive?

This chapter answers that question completely. `tensor_impl`, graph nodes, and backward traversal depend on this answer, so implementing them first would build on an undefined lifetime.

## 1.1 The Concrete Problem

Suppose we want this future API:

```cpp
auto x = tensor::from_vector(values, {2, 3}, true);
auto y = tensor::mul(x, x);
auto loss = tensor::sum(y);
loss.backward();
```

`tensor::mul` must still allocate an output and dispatch a Metal kernel. `loss.backward()` must later dispatch more kernels to compute gradients. Removing `ctx` from the call does not remove the runtime dependency; it only means the dependency must be found through another owner.

There are three obvious designs.

### Option A: Keep Passing `context&`

```cpp
auto y = tensor::mul(x, x, ctx);
loss.backward(ctx);
```

This remains correct and explicit. Its cost is repeated API plumbing: movement operations, composed operations, modules, optimizers, and future Python bindings all carry a context that is already determined by their tensor operands. It also leaves open what should happen when operands came from different contexts.

### Option B: Use One Global Singleton Context

```cpp
core::context& global_context();
```

Every operation could fetch this singleton. The common API would be short, but all tensors would silently depend on one process-wide object. Tests could not create isolated runtimes, mixed-runtime mistakes could not be represented and rejected, and a future change away from the singleton would touch the whole API.

### Option C: Bind Runtime Identity To Every Tensor And Provide A Default

Each tensor remembers the shared runtime that created it. Operations infer the runtime from their operands. Construction uses a process-wide default when the caller does not provide an explicit context.

```text
default context ----------- convenience for construction
explicit context ---------- optional isolated construction
tensor -------------------- owns the selected runtime identity
operation ----------------- reads runtime from tensor operands
autograd graph ------------ owned by result tensors, not by the default
```

We choose Option C. It gives the clean common API of a global context without making the singleton the only possible owner.

## 1.2 The Key Split: Handle Versus State

The word "context" currently refers to two responsibilities at once:

- It is the small object passed around by user code.
- It owns the heavy Metal resources used by operations.

We separate those responsibilities:

```text
core::context               small copyable public handle
  └── shared_ptr<runtime_state>
        ├── unique_ptr<metal_device>
        └── unique_ptr<kernel_library>
```

`context` becomes cheap to copy because it copies one `shared_ptr`. `runtime_state` is the actual resource owner. Several context handles and tensors may share the same runtime state, while a separately constructed context owns a different runtime state.

This is the same general pattern already used by `tensor` and `storage`: a small object shares a heavier owner. The important new detail is that runtime identity is the address of the shared `runtime_state`, not the address of a particular `context` handle.

## 1.3 Why `shared_ptr` Outside And `unique_ptr` Inside?

The ownership types describe two different relationships:

```text
many handles/tensors may share one runtime_state     -> shared_ptr
one runtime_state exclusively owns one device       -> unique_ptr
one runtime_state exclusively owns one kernel cache -> unique_ptr
```

Using `shared_ptr` for the device and kernel library too would be weaker: it would suggest that they may outlive or move independently from the runtime state. They should not. One runtime state is their obvious owner, so `unique_ptr` remains the better fit internally.

The resulting destruction order is also useful. `runtime_state` declares `device_` before `kernels_`, but C++ destroys members in reverse declaration order. The kernel library is therefore destroyed before the device it depends on.

## 1.4 A First Concrete Definition

Keep the first implementation direct. `runtime_state` can live in `comtam/core/context.h` until repetition or the public-header boundary earns another file.

```cpp
#pragma once

#include <memory>

#include "comtam/core/device.h"
#include "comtam/core/kernel.h"

namespace comtam::core {

class runtime_state {
   public:
    runtime_state();
    ~runtime_state() = default;

    runtime_state(const runtime_state&) = delete;
    runtime_state& operator=(const runtime_state&) = delete;

    metal_device& device() noexcept { return *device_; }
    const metal_device& device() const noexcept { return *device_; }

    kernel_library& kernels() noexcept { return *kernels_; }
    const kernel_library& kernels() const noexcept { return *kernels_; }

   private:
    std::unique_ptr<metal_device> device_;
    std::unique_ptr<kernel_library> kernels_;
};

class context {
   public:
    context();
    ~context() = default;

    context(const context&) = default;
    context& operator=(const context&) = default;

    metal_device& device() noexcept { return state_->device(); }
    const metal_device& device() const noexcept { return state_->device(); }

    kernel_library& kernels() noexcept { return state_->kernels(); }
    const kernel_library& kernels() const noexcept { return state_->kernels(); }

    const std::shared_ptr<runtime_state>& state() const noexcept { return state_; }

    bool shares_runtime_with(const context& other) const noexcept {
        return state_ == other.state_;
    }

   private:
    std::shared_ptr<runtime_state> state_;
};

COMTAM_INLINE context& default_context() {
    static context instance;
    return instance;
}

}  // namespace comtam::core
```

Read this definition from the inside out:

1. `runtime_state` cannot be copied because there must not be two C++ owners pretending to own the same Metal objects.
2. `context` can be copied because copying it only adds another reference to the same runtime state.
3. Move operations are not added yet, so an ordinary constructed or copied context never becomes a moved-from handle with null state.
4. Existing callers still use `ctx.device()` and `ctx.kernels()`, so this refactor need not change tensor dispatch yet.
5. `state()` exposes the shared runtime identity that `tensor_impl` will store in Chapter 2.
6. `shares_runtime_with` expresses identity without exposing pointer comparison throughout the codebase.

The public `state()` accessor is acceptable during the learning implementation. Module 10 can later hide it behind internal headers or friendship after the ownership model is stable.

## 1.5 Constructing The Shared State

The implementation moves the current context construction into `runtime_state`. The default accessor remains inline in `context.h`, as shown above, while the resource-owning constructors live in `context.cpp`:

```cpp
#include "comtam/core/context.h"

#include <memory>

#ifndef COMTAM_KERNEL_DIR
#define COMTAM_KERNEL_DIR "kernels"
#endif

namespace comtam::core {

runtime_state::runtime_state()
    : device_(std::make_unique<metal_device>()),
      kernels_(std::make_unique<kernel_library>(device_->get(),
                                                COMTAM_KERNEL_DIR)) {}

context::context() : state_(std::make_shared<runtime_state>()) {}

}  // namespace comtam::core
```

The constructor order matters. `device_` is constructed first, so `device_->get()` is valid when `kernel_library` is constructed. Do not reverse those declarations or initializer dependencies.

An inline function with a function-local static still denotes one shared static object across the program. It gives us three useful properties:

- It is initialized the first time `default_context()` is called rather than during unrelated static initialization.
- C++20 guarantees thread-safe initialization of the function-local static.
- Every call returns the same context handle and therefore the same runtime state.

The default context is not replaceable in Module 6. A mutable `set_default_context` would introduce synchronization and lifetime questions without helping the current single-device goal.

## 1.6 Why Tensor Lifetime Will Remain Safe

Later, `tensor_impl` will copy `ctx.state()`:

```cpp
struct tensor_impl {
    std::shared_ptr<core::runtime_state> runtime;
    // dtype, view, storage, and autograd metadata come later.
};
```

Consider an explicit context that goes out of scope while a tensor survives:

```cpp
tensor make_tensor() {
    core::context local;
    return tensor::from_vector(values, shape, local);
}  // local context handle is destroyed here
```

The tensor's `shared_ptr<runtime_state>` keeps the state alive. The Metal device, queue, kernel library, and tensor storage remain valid until the final tensor, context handle, or other runtime owner releases the state.

The reverse relationship must never exist:

```text
runtime_state -> tensor
```

If the process-wide default runtime owned tensors or graph nodes, those objects would survive until process shutdown. The runtime owns execution resources; tensors own graph reachability.

## 1.7 How Operations Will Infer A Runtime

Chapter 1 does not remove context parameters from tensor operations yet. It only creates the ownership mechanism that makes removal possible. After `tensor_impl` exists, inference will follow these rules:

```text
unary or movement op
  -> use input.impl_->runtime

binary op
  -> require lhs.impl_->runtime == rhs.impl_->runtime
  -> use that shared runtime

scalar constant inside mean/div/etc.
  -> construct it in the non-scalar operand's runtime

backward()
  -> use the scalar root's runtime
```

Different runtime identities must fail before allocation or dispatch. Comtam will not silently copy a tensor between runtimes. A future transfer operation must be explicit because it creates new storage and a new ownership identity.

This is why the global default is only a construction convenience. Once a tensor exists, operations use the tensor-bound runtime and do not repeatedly consult global state.

## 1.8 Where `no_grad` Will Live

It is tempting to add this member immediately:

```cpp
class runtime_state {
    bool no_grad_ = false;  // do not do this
};
```

That Boolean would be shared by every thread using the runtime. A `no_grad` scope in one Python or C++ thread would unexpectedly disable recording in another.

The accepted design instead makes recording suppression thread-local and keyed by runtime identity. The runtime state supplies the identity, but it does not store one shared recording Boolean. Chapter 3 will build the nesting-depth mechanism after tensors can carry runtime identity.

This distinction is worth remembering:

```text
runtime resources       shared across tensor handles and threads
recording mode          local to one thread and one runtime
autograd graph          owned by tensor results
```

No tape belongs in `runtime_state`.

## 1.9 Implementation Checkpoints

Do not combine the runtime refactor with `tensor_impl` in one unverified edit. Use these checkpoints:

1. Add `runtime_state` and make `context` delegate `device()` and `kernels()` to it.
2. Build and run the existing forward tests without changing tensor APIs.
3. Add copy and identity tests for `context`.
4. Only after that passes, begin Chapter 2 and let `tensor_impl` retain `ctx.state()`.

The first checkpoint should be behavior-preserving. If an existing forward test changes result, the runtime refactor is wrong; autograd is not involved yet.

## 1.10 Tests That Prove The Design

Start with identity tests:

```cpp
TEST_CASE("copied contexts share one runtime") {
    core::context a;
    core::context b = a;

    REQUIRE(a.shares_runtime_with(b));
    REQUIRE(&a.device() == &b.device());
    REQUIRE(&a.kernels() == &b.kernels());
}

TEST_CASE("separately constructed contexts are isolated") {
    core::context a;
    core::context b;

    REQUIRE_FALSE(a.shares_runtime_with(b));
}

TEST_CASE("default context is stable") {
    auto& a = core::default_context();
    auto& b = core::default_context();

    REQUIRE(&a == &b);
    REQUIRE(a.shares_runtime_with(b));
}
```

Then rerun the existing context, tensor, and operator tests. They are regression evidence that moving ownership behind shared state did not alter Metal execution.

A stronger lifetime test belongs in Chapter 2, when `tensor_impl` actually retains the runtime state. Do not write a fake lifetime test before the ownership edge exists.

## 1.11 Common Mistakes

### Storing A Raw Context Pointer In Tensor

```cpp
core::context* context_;  // unsafe lifetime dependency
```

The caller can destroy the context handle while the tensor survives. Store the shared runtime state instead.

### Making `runtime_state` Copyable

Copying an object with exclusive Metal-resource ownership either fails to compile because of `unique_ptr` or tempts you to invent unclear copy semantics. Share the state; do not copy the state.

### Looking Up The Default Runtime On Every Operation

That would make a tensor's behavior depend on whatever the process currently considers default. Infer runtime from operands after construction.

### Putting The Autograd Graph In The Runtime

This turns runtime lifetime into graph lifetime and requires manual tape clearing. Results will own their producer nodes instead.

### Adding Thread-Local Recording Before Runtime Identity Exists

Recording mode must be keyed by a stable runtime identity. Implementing it first either creates a process-wide flag or commits to an identity mechanism accidentally.

## 1.12 Your Implementation Exercise

Implement only the runtime split before proceeding to `tensor_impl`:

1. Draw the current context ownership graph from `context.h` and `context.cpp`.
2. Add `runtime_state` with exclusive device and kernel-library ownership.
3. Make `context` a copyable shared handle without changing existing call sites.
4. Add `default_context()` but do not migrate tensor constructors or operations to it yet.
5. Add the three identity tests above.
6. Run the existing build and CTest suite.
7. Explain, in your own words, why destroying a copied context handle does not destroy the Metal device while another copy survives.

Do not begin Chapter 2 until the forward suite still passes and you can answer the ownership question without referring back to this page.

## 1.13 Chapter Completion Gate

Chapter 1 is complete when all of these statements are true:

- `context` is a cheap copyable handle.
- One `runtime_state` obviously owns one device and one kernel library.
- Copied contexts share runtime identity; separately constructed contexts do not.
- The default context is lazy, stable, and non-replaceable.
- Existing forward behavior and tests remain unchanged.
- At the Chapter 1 checkpoint, no tensor uses the default runtime implicitly; Chapter 2 may then make the stable default a construction convenience.
- No recording flag or autograd graph has been placed in the runtime state.

## Agent Feedback / Grading

Status: implementation complete; direct context-identity evidence incomplete.

The current source has the intended ownership split: `context` is a copyable handle over `shared_ptr<runtime_state>`, and `runtime_state` exclusively owns the device and kernel library in dependency-safe declaration order. Tensor lifetime tests also prove that an explicitly selected context handle may be destroyed while its tensor remains usable for later Metal operations.

The independently rebuilt Metal-backed suite passes 51/51 tests. The new tensor identity tests provide strong evidence for runtime isolation and lifetime, and repeated default-constructed tensors indirectly exercise one stable default runtime. However, `tests/core/context.cpp` still lacks the chapter's direct assertions that a copied context shares its device, kernels, and runtime state and that repeated `default_context()` calls return the same handle. Add those narrow assertions before recording Chapter 1 as passed.

Verification performed on the current revision:

```text
cmake --build build --target comtam_tests -j 6  -> compiled
ctest --test-dir build --output-on-failure      -> passed, 51/51 outside the sandbox
git diff --check                                -> passed
```

## What Comes Next

The next chapter will start from a similarly concrete question:

> If `tensor b = a` must share gradient identity, but `auto b = a.reshape(...)` must create a new identity while sharing storage, which object represents the logical tensor?

Continue with [`Chapter 2: Tensor Identity And tensor_impl`](02_TENSOR_IDENTITY.md). It builds `tensor_impl` from the current `tensor` fields, shows constructor migration, explains copy and movement behavior, and adds identity tests before introducing a single gradient rule.

Later chapters will use the same pattern:

- Start from one failure or ownership problem visible in comtam.
- Compare plausible designs and reject them for concrete reasons.
- Define the minimum C++ types needed for the chosen design.
- Trace construction, copying, destruction, and failure behavior.
- Implement one small checkpoint.
- Add tests that prove the mechanism rather than merely exercising it.

## Reflection Before Chapter 2

Before expanding the guide, evaluate this chapter against three questions:

1. Can you now explain why `runtime_state` exists instead of putting everything directly in a global `context`?
2. Could you implement the runtime split by following the checkpoints without copying code you do not understand?
3. Is any transition still too large, especially the jump from shared runtime state to the future tensor-bound runtime?

Your answers will calibrate the `tensor_impl` chapter. The next chapter should not become shorter or more abstract than this one.
