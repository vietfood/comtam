---
name: comtam-metal
description: Implement or review comtam's Metal and metal-cpp host code, especially object ownership, errors, command submission, resource binding, dispatch, and synchronization.
---

# Comtam Metal work

Use the current source as the architectural authority. Before changing Metal
code, read `AGENTS.md`, `README.md`, and the host and kernel paths involved in
the request. In particular, inspect `comtam/core/device.*`,
`comtam/core/kernel.*`, `comtam/core/storage.*`, and the matching
`comtam/kernels/*.metal` entry points as relevant.

## Preserve these invariants

- Keep long-lived owned Metal objects in `NS::SharedPtr`.
- Wrap returns owned under Cocoa naming rules (`alloc`, `new`, `copy`,
  `mutableCopy`, and `Create`) with `NS::TransferPtr`.
- Treat `NS::Error*` returned through an error parameter as borrowed
  diagnostic data. Do not release it.
- Use `NS::RetainPtr` only when a borrowed object must outlive its current
  scope. Avoid manual `release()` in ordinary project code.
- Put an autorelease pool around scopes that create autoreleased Foundation or
  Metal objects. Do not assume metal-cpp gives Objective-C objects ordinary
  C++ ownership semantics.
- Report unrecoverable setup and execution failures as C++ exceptions, using
  the project's existing checks and `comtam::utils::ns_error_message` where
  appropriate.
- On the current synchronous path, check command-buffer status after
  `commit()` and `waitUntilCompleted()`.
- Keep synchronous execution until a requested change defines the lifetime,
  synchronization, and failure semantics of an asynchronous replacement.

## Change workflow

1. Classify each Metal object touched by the change as owned or borrowed and
   identify the scope that controls its lifetime.
2. Trace buffer and byte binding indices between host dispatch and the exact
   Metal function signature. Keep the two sides in one focused change.
3. Derive dispatch geometry from the kernel contract. Distinguish threads from
   threadgroups and do not substitute `threadExecutionWidth()` for a fixed
   tile or reduction shape.
4. Keep the single-device, shared-buffer, float32-first project scope unless
   the request explicitly changes it.
5. Add or update a correctness test for semantic changes. Use an independent
   numerical oracle where applicable; printed output is only a diagnostic.
6. Run `./build.sh` and `./test.sh`. Report unavailable Metal execution or
   unrun commands precisely.

Prefer a direct implementation over a new manager, backend layer, cache, or
asynchronous abstraction unless repetition or measurement demonstrates the
need.
