## Module 1: Storage And Tensor Metadata

Settle ownership before you settle behavior.

In an eager Metal framework, the first thing that will hurt you is not wrong math.
It is unclear lifetime: a buffer freed while a command still references it, two
tensors that think they own the same memory, or a copy that should have been a
share. This module makes ownership boring and explicit so the rest of the course
can move fast.

## Why This Module Exists

Storage is the boundary between C++ lifetime and GPU memory.

Every later module assumes that a `Tensor` can be copied, sliced, and passed into
a kernel without anyone guessing who frees the underlying `MTL::Buffer`. Views,
dispatch, autograd, and training all build on top of that guarantee.

The main design decision in this module is to keep one rule from
[`../ARCHITECTURE.md`](../ARCHITECTURE.md):

```text
one Storage owns one buffer
tensors may share storage
views do not allocate
```

That is deliberately smaller than ATen's storage system. See
[`../AVOID.md`](../AVOID.md) for why custom deleters and context pointers are
postponed.

## Mental Model

Separate the three layers and never let them blur:

```text
MTL::Buffer        raw GPU memory, reference counted by metal-cpp
Storage            one C++ owner of one MTL::Buffer + size in bytes
Tensor             DType + View + a shared_ptr<Storage>
```

Why three layers and not two:

- `MTL::Buffer` is an Objective-C object. Its lifetime follows Cocoa rules and is
  managed by `NS::SharedPtr`.
- `Storage` gives that buffer a single, obvious C++ owner and a byte size, so the
  rest of comtam never touches metal-cpp retain/release directly.
- `Tensor` is just an interpretation. Many tensors can point at one `Storage`
  through `std::shared_ptr`.

The current code already encodes this:

```text
comtam/core/storage.h   Storage: size_ + NS::SharedPtr<MTL::Buffer>, move-only
comtam/tensor.h         Tensor: dtype_, View view_, shared_ptr<Storage> storage_
```

## Suspicious Assumptions To Test

Do not treat this list as known bugs. Treat it as assumptions the code must
prove. Write a small probe for each before changing anything.

1. Is `Storage` really move-only, and does moving it leave the source safe to
   destroy? (See the move constructor and deleted copy in `storage.h:16`.)
2. When two tensors share a `Storage`, does the buffer survive until the last
   tensor dies?
3. Does `Device::copy` validate that byte counts match before `memcpy`?
   (See `device.h:36` and `device.h:45`.)
4. Does `Storage` know its element count, or only its byte size? Who converts
   between them, and using which `DType`?
5. Is there any path where a `Tensor` holds a null `storage_`? What is supposed
   to happen then? (See the null checks in `tensor.cpp:32` and `tensor.cpp:44`.)

When a probe surprises you, add it to [`BUG_LEDGER.md`](BUG_LEDGER.md).

## Learning Goals

By the end of this module you should be able to explain:

1. The difference between a buffer, a storage, and a tensor.
2. Why `Storage` is move-only and what that prevents.
3. Why tensors share `Storage` through `std::shared_ptr` instead of owning it.
4. How a byte count becomes an element count and back, and where `DType` enters.
5. What "views do not allocate" means in terms of `Storage` lifetime.
6. Why a single obvious owner per Metal object matters more than it looks.

## Assignment 1.1: Trace Storage Lifetime ⭐

**Task:** Trace the lifetime of one `MTL::Buffer` from allocation to release.

```cpp
Tensor t(A.data(), {N}, device);
```

**Questions to answer:**

1. `Device::allocate(bytes)` returns a `Storage` by value. How does the move
   constructor in `storage.h:16` avoid a double free of the buffer?
2. The constructor wraps the storage in `std::make_shared<Storage>(...)`
   (`tensor.cpp:17`). After this line, how many owners does the `Storage` have?
   How many owners does the `MTL::Buffer` have?
3. When `t` goes out of scope, what is the exact destruction order:
   `Tensor` -> `shared_ptr` -> `Storage` -> `NS::SharedPtr` -> `MTL::Buffer`?

**Deliverable:** An ownership diagram annotated with where each reference count
increments and decrements.

**Why this assignment:** Lifetime bugs are silent until a kernel reads freed
memory. Tracing the happy path first makes the unhappy paths visible later.

## Assignment 1.2: Make Element Count Explicit ⭐⭐

**Observation:** `Storage` stores `size_` in bytes. `Tensor` stores element
count through `View::numel()`. The conversion uses `dtype_size_map`
(`dtype.h:11`).

**Task:** Decide and document where byte/element conversion lives, then make it
consistent.

Questions:

1. Should `Storage` know its `DType`, or stay a dumb byte buffer?
2. Today `Device::copy` checks `count * sizeof(T) == storage.size()`
   (`device.h:38`). Is `sizeof(T)` from the call site guaranteed to match the
   tensor's `DType`? Where could they diverge?
3. If you add a second dtype later, which of these checks silently breaks?

**Recommended first choice:** Keep `Storage` a dumb byte buffer. Put all
dtype-aware sizing in `Tensor` and `Device::copy`. Do not give `Storage` a
`DType` until a second dtype forces it. This keeps the storage layer boring, in
the spirit of [`../AVOID.md`](../AVOID.md).

**Test to write:** A round-trip test that constructs a tensor, reads it back, and
asserts exact float equality, plus a negative test that a mismatched byte count
throws.

## Assignment 1.3: Prove Sharing Works ⭐⭐

**Task:** Demonstrate that two tensors can share one `Storage` safely.

You do not have view ops yet (that is Module 2), so simulate sharing directly:
construct two `Tensor` headers that hold the same `std::shared_ptr<Storage>`.

Questions:

1. If tensor `a` is destroyed, can tensor `b` still read its data?
2. If you write through `a`'s storage, does `b` see the change? Should it?
3. What invariant must hold for shared storage to be safe while a kernel is
   in flight? (Hint: `Device::submit` currently calls `waitUntilCompleted`.)

**Deliverable:** A small test that builds two tensors over one storage, destroys
one, and reads the other. Assert the survivor still returns correct data.

**Why this assignment:** Sharing is the whole point of `shared_ptr<Storage>`. If
you cannot prove it is safe now, views in Module 2 will be built on a guess.

## Assignment 1.4: Storage Debug Print ⭐

**Task:** `Storage::print(const std::string& label)` is declared in
`storage.h:38`. Implement or review it so it dumps buffer contents for debugging.

Rules:

- This is a debugging aid, not a test. Printing is never a correctness check.
- It must not change buffer contents or storage mode.
- Keep it small. Do not grow it into a logging framework
  (see [`../AVOID.md`](../AVOID.md) item 8).

**Why this assignment:** A tiny, honest print helper saves hours in later
modules, as long as you remember it proves nothing on its own.

## Assignment 1.5: Read Magnetron Storage Carefully ⭐⭐⭐

Read Magnetron's tensor and storage ownership after your own attempt.

Questions to answer:

1. How does Magnetron separate storage from tensor metadata?
2. What does Magnetron's reference counting do that `shared_ptr<Storage>` does
   not?
3. Which parts of Magnetron's ownership machinery are needed because it supports
   more than one backend?
4. Which one idea would you copy into comtam now, and which would you postpone?

**Deliverable:** A short note titled "What comtam should copy from Magnetron
storage, and what it should postpone."

## Module 1 Checklist

- [ ] 1.1 Trace one buffer's full lifetime with reference counts.
- [ ] 1.2 Decide where byte/element conversion lives; add round-trip tests.
- [ ] 1.3 Prove two tensors can share one storage safely.
- [ ] 1.4 Implement or review `Storage::print` as a debug aid only.
- [ ] 1.5 Read Magnetron storage and write a copy/postpone note.

## Exit Criteria

You are ready for Module 2 when:

1. Every Metal object has exactly one obvious C++ owner you can name.
2. You can explain why `Storage` is move-only and why `Tensor` shares it.
3. Round-trip and sharing tests pass with exact float comparisons.
4. Byte vs element conversion has one documented home.
5. You have a written decision on whether `Storage` should ever know its dtype.
