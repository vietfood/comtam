## Module 1: Storage Invariants

Settle storage before views.

In an eager Metal framework, the first thing that will hurt you is not wrong
math. It is unclear lifetime and unclear sizing: a buffer freed while a command
still references it, two tensors that accidentally copy when they should share,
or a byte count that silently stops matching the tensor's dtype.

This module is intentionally short. You already know what a tensor is. The goal
is to prove the invariants that Module 2 views will depend on.

## Core Rule

Keep this rule from [`../ARCHITECTURE.md`](../ARCHITECTURE.md):

```text
one Storage owns one buffer
tensors may share storage
views do not allocate
```

That is deliberately smaller than ATen's storage system. Do not add custom
deleters, backend registries, allocator caches, or dtype-aware storage unless a
failing test forces the design to grow. See [`../AVOID.md`](../AVOID.md).

## Mental Model

Keep the three layers separate:

```text
MTL::Buffer        raw GPU memory, reference counted by metal-cpp
Storage            one C++ owner of one MTL::Buffer + size in bytes
Tensor             DType + View + shared_ptr<Storage>
```

- `MTL::Buffer` is an Objective-C object managed through `NS::SharedPtr`.
- `Storage` gives the buffer one obvious C++ owner and records byte size.
- `Tensor` is an interpretation of storage. Multiple tensors may share the same
  `Storage`.

The current code should keep matching this shape:

```text
comtam/core/storage.h   Storage: size_ + NS::SharedPtr<MTL::Buffer>, move-only
comtam/tensor.h         Tensor: dtype_, View view_, shared_ptr<Storage> storage_
```

## Suspicious Assumptions To Test

Do not treat this list as known bugs. Treat it as assumptions the code must
prove:

1. Does `Device::copy` reject byte-count mismatches before `memcpy`?
2. Does byte/element conversion have one home, or is it spread across call sites?
3. Can two tensors share one `Storage` and keep the buffer alive until the last
   tensor dies?
4. If one tensor writes through shared storage, does the other tensor observe the
   change?
5. Is `Storage::print` honest about being float32-only debug output?

When a probe surprises you, record it in your solution note under the relevant
assignment feedback.

## Assignment 1.1: Make Byte Sizing Explicit ⭐⭐

**Observation:** `Storage` stores `size_` in bytes. `Tensor` stores element count
through `View::numel()`. The conversion currently depends on `dtype_size_map`.

**Task:** Decide where byte/element conversion lives, then make the code and
tests enforce that decision.

Questions:

1. Should `Storage` know its `DType`, or stay a dumb byte buffer?
2. Is `sizeof(T)` in `Device::copy<T>` guaranteed to match the tensor's `DType`?
3. If a second dtype is added later, which checks would silently become wrong?

**Recommended first choice:** Keep `Storage` a dumb byte buffer. Put dtype-aware
sizing at the tensor/copy boundary. Do not give `Storage` a `DType` until a
second dtype forces it.

**Tests to write:**

- A float32 round-trip test that asserts exact equality.
- A negative test that a mismatched byte count throws.

## Assignment 1.2: Prove Sharing Works ⭐⭐

**Task:** Demonstrate that two tensor headers can share one `Storage` safely.

You do not need full view ops yet. Simulate sharing directly by constructing two
tensor headers over the same `std::shared_ptr<Storage>`, or add the smallest test
helper needed to do that without weakening the public API.

Questions:

1. If tensor `a` is destroyed, can tensor `b` still read the data?
2. If you write through `a`'s storage, does `b` see the change? Should it?
3. What lifetime invariant must hold while a kernel is in flight?

Hint: `Device::submit` currently calls `waitUntilCompleted`, so the first
version can stay synchronous.

**Test to write:** Build two tensors over one storage, destroy one, and assert
the survivor still returns correct data.

## Assignment 1.3: Keep `Storage::print` Small ⭐

**Task:** Implement or review `Storage::print(const std::string& label)`.

Rules:

- It is a debugging aid, not a test.
- It must not change buffer contents or storage mode.
- It may assume float32 for now, but that assumption must be obvious.
- It must not grow into a logging framework.

Printed arrays prove nothing. They are useful only when paired with tests that
have oracles.

## Optional Reading: Magnetron Storage

Read Magnetron's storage and tensor ownership after your own tests pass, not as
a blocker before Module 2.

Questions worth answering:

1. How does Magnetron separate storage from tensor metadata?
2. What does Magnetron's reference counting do that `shared_ptr<Storage>` does
   not?
3. Which parts exist mainly because Magnetron supports more than one backend?
4. Which one idea would you copy into comtam now, and which would you postpone?

## Module 1 Checklist

- [ ] 1.1 Byte/element conversion has one documented home.
- [ ] 1.1 Round-trip and byte-mismatch tests pass.
- [ ] 1.2 Shared storage lifetime is tested.
- [ ] 1.2 Shared writes/readback behavior is understood and documented.
- [ ] 1.3 `Storage::print` is implemented or reviewed as debug-only.

## Exit Criteria

You are ready for Module 2 when:

1. Every Metal buffer has one obvious C++ owner you can name.
2. `Storage` remains move-only and dtype-agnostic unless a test proves otherwise.
3. Round-trip and sharing tests pass with exact float comparisons.
4. Byte vs element conversion has one documented home.
5. Debug printing is clearly separated from correctness testing.
