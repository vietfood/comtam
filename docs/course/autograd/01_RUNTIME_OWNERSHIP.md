# Chapter 1: Runtime Ownership Before Autograd

**Status:** Implemented. Gate met.

This chapter records a decision rather than setting an assignment. The runtime split it describes is already in the source; read it to understand why `context` and `runtime_state` are two types, because every later chapter assumes that answer.

## The Concrete Problem

Before this chapter, every operation reached Metal through a `context&` that the caller supplied and kept alive. That works while a human writes each call. It stops working the moment a backward rule needs a device.

A backward rule runs inside the engine, not inside user code. When a future `mul_backward` needs to launch a kernel, nobody is standing there to hand it a context. It has to recover the runtime from the tensors it was given. So the question is:

> Where do operations obtain Metal resources, and who guarantees those resources outlive every tensor that depends on them?

Three answers were plausible.

**Keep passing `context&`.** Every op signature grows a parameter, every composition threads it through, and backward rules would need the context stored somewhere anyway - so this only moves the problem. Worse, a reference gives no lifetime guarantee at all. A tensor holding `context&` dangles the moment the caller's context leaves scope.

**Use one global singleton context.** Convenient, and wrong for the reason [`docs/note/AVOID.md`](../../note/AVOID.md) records in item 2: a global Metal manager makes runtime identity untestable. Two tests can no longer construct isolated devices, and "which device did this tensor come from" stops being a question the type system can answer.

**Bind runtime identity to every tensor, with a process-wide default for convenience.** Each tensor knows its own runtime and keeps it alive. Operations infer the runtime from their operands. Construction may fall back to a shared default so ordinary code stays short. This is what comtam does.

## The Key Split: Handle Versus State

The third answer needs two types, because one type would have to be both cheap to copy and the sole owner of expensive Metal objects. Those requirements conflict.

`runtime_state` ([context.h:20](../../../comtam/core/context.h:20)) is the owner. It holds the device and the kernel library in `unique_ptr`s and is non-copyable. One `runtime_state` means one device and one library, with no ambiguity about who releases them.

`context` ([context.h:39](../../../comtam/core/context.h:39)) is the handle. It holds a `shared_ptr<runtime_state>` and is freely copyable. Copying a context adds a reference; it never duplicates a device.

Read `shared_ptr` outside and `unique_ptr` inside as a deliberate pair:

- Outside, ownership is genuinely shared. A tensor, a context handle, and a future graph node can all legitimately keep the same runtime alive, and none of them can know which will die last. That is what `shared_ptr` is for.
- Inside, ownership is not shared. Exactly one `runtime_state` owns the device. `unique_ptr` says so in the type and costs no atomic refcount.

Order is load-bearing twice over, in two different places. `kernel_library` is built from the device, so the constructor body assigns `device_` before `kernels_` ([context.cpp:22](../../../comtam/core/context.cpp:22)); swapping those two statements is a use-before-init that compiles cleanly. Separately, `device_` is *declared* before `kernels_` ([context.h:35](../../../comtam/core/context.h:35)), which is what makes the device outlive the library at destruction. Declaration order and assignment order are doing different jobs here; changing either one alone breaks a different invariant.

`context` deliberately has no move operations. Without them, an ordinary constructed or copied context can never be a moved-from handle with null state, so no accessor needs a null check.

## The Default Context

`default_context()` ([context.h:63](../../../comtam/core/context.h:63)) is an inline function wrapping a function-local static. That buys three properties worth naming:

- It initializes on first use, not during static initialization of some unrelated translation unit.
- C++20 guarantees the initialization is thread-safe.
- Every call returns the same handle, therefore the same runtime state.

It is not replaceable. A `set_default_context` would raise synchronization and lifetime questions - what happens to tensors built from the old default? - with no benefit to a single-device framework.

## Why Tensor Lifetime Is Safe

`tensor_impl` copies `context::state()` rather than referencing the context ([impl.h:39](../../../comtam/tensor/impl.h:39)). The consequence is the whole point of the design:

```text
context handle (scope-local)  --shared_ptr-->  runtime_state  <--shared_ptr--  tensor_impl
```

Both arrows own. Destroying the context handle decrements one reference; if a tensor still holds the other, the device stays alive and the tensor stays usable. [`tests/tensor/identity.cpp:154`](../../../tests/tensor/identity.cpp:154) constructs a tensor from an explicit context, destroys the context, then runs a Metal operation on the survivor.

## What Was Deliberately Left Out

No recording flag and no autograd graph went into `runtime_state`. The temptation is real: the runtime is reachable from everywhere, which makes it an attractive place to hang a global `grad_enabled` bool. That is precisely why to refuse. It would be hidden global framework state under a different name, and it would make recording suppression process-wide when it should be scoped per thread and per runtime. Chapter 3 places recording state deliberately instead.

## Common Mistakes

**Storing a raw `context*` or `context&` in a tensor.** Restores the dangling problem this chapter set out to remove. A tensor must own a reference to the *state*, not point at somebody's handle.

**Making `runtime_state` copyable.** Two owners of one device, and a double release when both die.

**Looking up `default_context()` inside an operation.** Operations infer the runtime from their operands. Consulting the default would silently repair mixed-runtime bugs instead of rejecting them. The default is a *construction* convenience only.

**Reversing the `device_`/`kernels_` declaration order.** Compiles, then reads an uninitialized `unique_ptr`.

## Chapter Gate

All of these hold in the current source:

- `context` is a cheap copyable handle; `runtime_state` is the non-copyable owner.
- One `runtime_state` owns one device and one kernel library, in dependency-safe order.
- Copied contexts share runtime identity; separately constructed contexts do not.
- The default context is lazy, stable, and non-replaceable.
- Forward behavior and existing tests are unchanged.
- No recording flag or autograd graph lives in the runtime state.

## Agent Feedback / Grading

**Status: passed.**

The ownership split is implemented as designed. `context` is a copyable handle over `shared_ptr<runtime_state>`; `runtime_state` exclusively owns the device and kernel library in dependency-safe declaration order.

The direct context-identity assertions that this chapter previously listed as missing are now present. [`tests/core/context.cpp:22`](../../../tests/core/context.cpp:22) asserts that a copied context shares runtime state, device, and kernel library; [`tests/core/context.cpp:31`](../../../tests/core/context.cpp:31) asserts that separately constructed contexts do not; [`tests/core/context.cpp:38`](../../../tests/core/context.cpp:38) asserts that repeated `default_context()` calls return the same handle. Tensor lifetime across context destruction is covered by [`tests/tensor/identity.cpp:154`](../../../tests/tensor/identity.cpp:154).

Verification recorded at grading time:

```text
cmake --build build --target comtam_tests -j 6  -> compiled
ctest --test-dir build --output-on-failure      -> passed, 51/51 outside the sandbox
```

## What Comes Next

[Chapter 2](02_TENSOR_IDENTITY.md) answers the question this ownership model raises immediately: if a tensor owns runtime identity, what owns *tensor* identity, and how can `tensor b = a` share a gradient while `a.reshape(...)` does not?
