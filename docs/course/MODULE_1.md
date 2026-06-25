# Module 1: Metal Ownership And Storage

The first real framework problem is ownership. If this is sloppy, every later
bug will look like a tensor bug while actually being a lifetime bug.

## Learning Goals

By the end of this module, you should be able to explain:

1. Who owns the `MTL::Device`.
2. Who owns the `MTL::CommandQueue`.
3. Who owns each `MTL::Buffer`.
4. When host data is copied into a buffer.
5. When buffer contents are read back to the host.

## Minimal Types

Implement:

```cpp
class Context;
class Device;
class Storage;
```

`Storage` should own:

- `MTL::Buffer*`
- byte size
- storage mode

Start with `MTL::ResourceStorageModeShared`. Private buffers can wait.

## Anti-Pattern To Avoid

Do not copy legrad's early allocator design:

- no global Metal singleton
- no comparable deleters
- no bucket allocator
- no profiling counters in `Device`

Those are tools for a mature runtime. In a tiny framework, they are fog.

## Assignment 1.1: RAII Wrappers

Write a tiny move-only wrapper for Objective-C reference-counted Metal objects.

Required behavior:

- calls `release()` in the destructor
- supports move construction
- supports move assignment
- does not support copy
- exposes `get()`

## Assignment 1.2: Storage Allocation

Add:

```cpp
Storage Device::allocate(size_t bytes);
```

Reject zero-byte allocations unless you deliberately support scalar-only
storage.

## Assignment 1.3: Host Roundtrip

Write tests for:

1. Allocate a buffer.
2. Copy a `std::vector<float>` into it.
3. Copy it back.
4. Compare exact values.

## Completion Gate

You may start Module 2 only when:

- `Storage` roundtrips host data correctly.
- Metal objects have clear single ownership.
- No global singleton exists.
- No caching allocator exists.

