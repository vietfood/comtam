# Why legrad collapsed and what comtam must not repeat

This document is not a list of generic best practices. It is a guardrail for
`comtam`, written after `legrad` became too complex too quickly.

The rule is simple:

```text
Do not import a big-framework abstraction before comtam has the pressure that
made that abstraction necessary.
```

## 1. ATen-shaped complexity before an end-to-end loop

### What happened in legrad

`legrad` copied ideas from ATen early: allocator interfaces, raw buffer
contexts, custom deleters, and backend-flavored ownership machinery.

Those ideas are not bad. They are just expensive.

ATen needs them because PyTorch supports many devices, storage backends, Python
interop, dispatch keys, view semantics, graph capture, custom memory formats,
legacy behavior, and production compatibility. `comtam` does not.

### Why it hurt

The framework began paying design rent before it had revenue:

- no stable tiny training loop
- unclear tensor ownership
- allocator complexity before memory pressure
- backend abstractions before multiple backends
- too many places to ask "who owns this?"

### comtam rule

First make this work:

```text
Tensor a, b -> GPU add -> read back -> scalar loss -> backward -> SGD step
```

Only then add abstractions where repetition or measurement forces them.

## 2. Global Metal manager

### What happened in legrad

`legrad` used a singleton-style Metal manager that owned the device, command
queue, counters, and allocator.

### Why it hurt

A singleton makes small code feel convenient, but it hides lifetime and global
state:

- tests cannot easily create isolated contexts
- teardown order becomes implicit
- future multi-context behavior is awkward
- allocator policy gets coupled to device discovery
- every subsystem learns to reach for global state

### comtam rule

Use explicit ownership:

```cpp
comtam::Context ctx;
auto& device = ctx.device();
auto& kernels = ctx.kernels();
```

`Context` owns the runtime. Nothing important should require
`Manager::instance()`.

## 3. Custom allocator too early

### What happened in legrad

The Metal allocator had a bucket pool before the framework had a stable tensor
and autograd path.

### Why it hurt

Allocator bugs look like tensor bugs. They also create false design pressure:

- buffer reuse hides lifetime mistakes
- pool accounting becomes another subsystem
- allocation policy infects storage design
- performance work starts before correctness exists

### comtam rule

Start with direct Metal allocation:

```cpp
device.new_buffer(bytes)
```

Add an `Allocator` only when code repetition is real. Add caching only when a
benchmark says allocation dominates.

## 4. Comparable deleters and context pointers

### What happened in legrad

`RawBuffer` stored a pointer, a context pointer, and a deleter. It also had
machinery to compare deleters.

### Why it hurt

That design is useful in a PyTorch-like storage system with borrowed memory,
custom allocators, foreign runtimes, and many ownership cases.

For `comtam`, it creates a second memory model before the first one is stable.

### comtam rule

A first `Storage` should be boring:

```text
Storage
  MTL::Buffer*
  size_bytes
```

It owns the buffer. Views share `Storage`. That is enough until proven otherwise.

## 5. Backend registry before multiple backends

### What Magnetron does well

Magnetron has a clean backend/device/command split because it targets CPU and
CUDA-style execution and wants backend extensibility.

### What comtam should borrow

Borrow this:

```text
op stub -> Command -> Device::submit
```

Do not borrow this yet:

```text
dynamic backend registry
backend shared libraries
ABI cookies
multi-device discovery
```

### comtam rule

There is one backend: Metal.

Write the code so a future backend is not impossible, but do not build a
backend framework now.

## 6. Kernel abstraction before one kernel works

### Failure mode

It is tempting to design a beautiful kernel manager before `vadd` runs.

### Why it hurt

Kernel management has real complexity:

- source loading
- runtime compilation
- offline `.metallib`
- pipeline caching
- function constants
- error reporting
- argument binding

Doing all of that before a single kernel works makes the first milestone blurry.

### comtam rule

Early kernel management should be tiny:

```text
KernelLibrary
  load source or metallib
  pipeline(name)
```

No generic graph executor. No scheduling. No fusion.

## 7. Profiling before execution exists

### What happened in legrad

The Metal manager knew about GPU counters early.

### Why it hurt

Profiling is seductive because it feels systems-y. But before correctness, it
is noise.

### comtam rule

First:

```text
correct result
```

Then:

```text
simple command timing
```

Only later:

```text
GPU counters
```

## 8. Logging and macro infrastructure too early

### Why it hurts

Macros, logging wrappers, and assertion frameworks can become a second language
inside the codebase.

### comtam rule

Use ordinary C++ errors first:

- `std::runtime_error`
- small helper checks
- readable messages

Add logging only when there is enough runtime behavior to inspect.

## 9. Too many files before the shape is known

### Why it hurts

File boundaries feel like architecture. They are not. Early over-splitting makes
simple changes require many edits.

### comtam rule

Prefer colocated, obvious files:

```text
comtam/
  core/
    context.h
    context.cpp
    device.h
    device.cpp
    kernel.h
    kernel.cpp
    storage.h
    storage.cpp
```

Grow files when responsibilities become stable.

## 10. The final test

Before adding a subsystem, ask:

1. What bug or repetition forces this?
2. Can the current training loop work without it?
3. Does Magnetron have this because it is generally necessary, or because its
   scope is larger than ours?
4. Is this actually an ATen idea wearing a tiny-framework costume?

If the answer is vague, do not add it yet.
