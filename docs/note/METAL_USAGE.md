# Notes

These are project notes for decisions that are important enough to remember but
not yet stable enough to become architecture law.

## Metal-cpp ownership and errors

Metal is Objective-C first. `metal-cpp` gives C++ syntax, but it does not turn
Metal into ordinary C++ ownership.

The two rules to settle early:

1. how to report Metal errors
2. how to release Objective-C objects safely

## Error policy

For `comtam`, setup and execution failures should throw C++ exceptions.

Use `std::runtime_error` first. A custom `MetalError` can wait until ordinary
exceptions become repetitive.

This is fine because early `comtam` is not building a C API. If a Metal device
cannot be created, a shader cannot compile, a pipeline cannot be built, or a
command buffer fails, there is no useful local recovery yet.

Example:

```cpp
if (device == nullptr) {
  throw std::runtime_error("failed to create default Metal device");
}
```

For APIs that fill `NS::Error*`, convert the error into a readable C++ message.

```cpp
std::string ns_error_message(NS::Error* error) {
  if (error == nullptr) {
    return "unknown Metal error";
  }

  NS::String* description = error->localizedDescription();
  if (description == nullptr) {
    return "unknown Metal error";
  }

  return description->utf8String();
}
```

Usage:

```cpp
NS::Error* error = nullptr;
auto library = NS::TransferPtr(device->newLibrary(source, nullptr, &error));
if (!library) {
  throw std::runtime_error(
      "failed to compile Metal library: " + ns_error_message(error));
}
```

Do not release `NS::Error*`. Treat it as borrowed diagnostic information.

## Command buffer errors

After synchronous execution, check command buffer status.

```cpp
command_buffer->commit();
command_buffer->waitUntilCompleted();

if (command_buffer->status() == MTL::CommandBufferStatusError) {
  throw std::runtime_error(
      "Metal command buffer failed: " +
      ns_error_message(command_buffer->error()));
}
```

Blocking with `waitUntilCompleted()` is acceptable while correctness is still
being established. Async execution can wait until the synchronous path is
stable.

## Ownership policy

Use RAII for Objective-C objects.

The metal-cpp ownership rule follows Cocoa:

- names beginning with `alloc`, `new`, `copy`, `mutableCopy`, or `Create` return
  objects you own
- objects you own must eventually be released
- objects you do not own must not be released
- if you need to keep a borrowed object, retain it

Prefer `NS::SharedPtr<T>` over raw owning pointers.

Use `NS::TransferPtr(...)` when the API returns an object you already own:

```cpp
NS::SharedPtr<MTL::Device> device =
    NS::TransferPtr(MTL::CreateSystemDefaultDevice());

NS::SharedPtr<MTL::CommandQueue> queue =
    NS::TransferPtr(device->newCommandQueue());
```

Use `NS::RetainPtr(...)` only when keeping a borrowed object beyond the current
scope.

Do not call `release()` manually in ordinary `comtam` code unless there is a
specific reason. Manual release should be rare and localized.

## Device skeleton

A first `Device` can look like this:

```cpp
#pragma once

#include "Foundation/NSSharedPtr.hpp"
#include "Metal/Metal.hpp"

namespace comtam::core {

class Device {
 public:
  Device();

  MTL::Device* raw() const { return device_.get(); }
  MTL::CommandQueue* queue() const { return command_queue_.get(); }

 private:
  NS::SharedPtr<MTL::Device> device_;
  NS::SharedPtr<MTL::CommandQueue> command_queue_;
};

}  // namespace comtam::core
```

And:

```cpp
Device::Device() {
  device_ = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device_) {
    throw std::runtime_error("failed to create default Metal device");
  }

  command_queue_ = NS::TransferPtr(device_->newCommandQueue());
  if (!command_queue_) {
    throw std::runtime_error("failed to create Metal command queue");
  }
}
```

This is simpler than legrad's singleton manager while still respecting
Objective-C ownership.

## Autorelease pools

Some Metal and Foundation methods return autoreleased objects. Without an
autorelease pool, those objects can leak.

In Objective-C, Apple's sample uses:

```objc
@autoreleasepool {
  // Metal code
}
```

In metal-cpp, use:

```cpp
auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

// Metal code
```

Create an autorelease pool in `main`. If a future loop creates many temporary
Foundation or Metal objects, create a smaller pool inside the loop.

## Practical summary

For early `comtam`:

- long-lived Metal objects are owned by `NS::SharedPtr`
- use `NS::TransferPtr` for owned returns from `new`, `alloc`, and `Create`
- use `NS::RetainPtr` only to keep borrowed objects
- convert `NS::Error*` to `std::runtime_error`
- check command buffer status after `waitUntilCompleted()`
- create an autorelease pool in `main`
- avoid a global Metal manager
