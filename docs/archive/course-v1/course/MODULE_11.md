## Module 11: Execution, Synchronization, And Memory

Decide from measurement whether to remove host blocking, then preserve eager
semantics, resource lifetime, and observable errors in the selected design.

## Why This Comes After Semantic Stability

Today each submission waits for completion, which makes buffer and error
lifetime obvious. Async execution changes when work completes, when failures
surface, and how long inputs and temporary buffers must remain alive. Those are
semantic changes, so they belong after the public tensor contract is pinned.

## Module Contract

**Prerequisites:** Module 10 passes, and the Module 8/13-style baseline shows
that synchronization or allocation is relevant enough to investigate.

**You will produce:** a measured synchronous-versus-async decision and explicit
synchronization/error/lifetime semantics for the selected path. If async is
justified, you will also produce a command-lifetime owner, deferred Metal error
propagation, dependency-safe eager submission, and async stress tests. Buffer
caching is independently optional and requires an allocation measurement.

**Supported scope:** one Metal device and queue, eager op creation, shared Metal
buffers, and ordered submission.

**Not required:** multiple queues, cross-device events, graph compilation,
automatic fusion, or a general scheduler.

## Assignment 11.1: Measure The Blocking Baseline ⭐⭐

**Task:** measure a chain of small ops, a large op, host readback, and one
training step. Separate encode/submit time from wait time where Metal permits.

Do not change execution until the report identifies which waits are redundant
and estimates the expected benefit. Before measuring, define a minimum meaningful
improvement relative to variance.

This is the module's branch point:

- **Synchronous track:** if waits are not a material cost for the supported
  workload, retain blocking submission, document why, and run the common
  synchronization/error/lifetime and allocation checks below.
- **Async track:** if the expected improvement is material, complete Assignments
  11.2-11.4 and the async form of 11.6.

Do not implement async and then call it optional after ownership work fails. The
track is selected from the baseline, and any later rejected experiment restores
the synchronous contract with the rejection evidence recorded.

## Assignment 11.2: Define Synchronization Semantics ⭐⭐⭐

**Task:** specify exactly which calls wait for GPU completion in the selected
track. On the synchronous track, pin the current per-op wait and error boundary
with tests. On the async track, use the recommended first contract below.

Recommended first contract:

- tensor ops encode and submit in order but do not wait
- operations on the same queue rely on Metal ordering for dependencies
- host readback waits for the producing work
- `context::synchronize()` waits explicitly
- context destruction completes or safely cancels outstanding ownership
- the next synchronization boundary reports earlier command failures

The API must never return host values before their producer completes.

## Assignment 11.3: Own In-Flight Resources ⭐⭐⭐

**Async track only. Task:** introduce the smallest object that keeps command buffers, input/output
storage, temporary metadata buffers, and autorelease-managed objects alive until
completion.

Use completion handlers or an explicit in-flight queue to retire resources. Draw
the ownership timeline and prove that dropping all user tensor handles before
completion does not free GPU-visible storage prematurely.

## Assignment 11.4: Propagate Deferred Errors ⭐⭐⭐

**Async track only. Task:** retain operation context for each in-flight command so a later wait can
report which operation failed. Define whether the context becomes poisoned after
a command error and how subsequent calls behave.

**Tests:** safely induced missing/invalid kernel errors, explicit synchronize,
readback-triggered synchronize, and context teardown with pending work.

## Assignment 11.5: Decide Whether Allocation Needs A Cache ⭐⭐

**Task:** count allocations and measure their contribution during the Module 8
workload. If allocation is material, add a context-owned cache with exact size,
alignment, in-flight exclusion, and bounded-retention rules. If it is not
material, record the result and do not add a cache.

A rejected cache is a valid completion outcome. Correct lifetime is mandatory;
reuse is optional.

## Assignment 11.6: Stress The Selected Execution Model ⭐⭐⭐

**Task:** run long chains and training steps and compare results to the reference.
On the synchronous track, prove each completed call releases transient command
state and reports errors at that call. On the async track, avoid intermediate
waits, drop temporary handles aggressively, then synchronize and assert in-flight
queues and any cache return to configured bounds.

## Module 11 Checklist

- [ ] 11.1 Blocking baseline separates submission, wait, and readback costs.
- [ ] 11.1 Synchronous or async track is selected from a stated threshold.
- [ ] 11.2 Selected synchronization and context-teardown behavior is tested.
- [ ] 11.3 Async track: in-flight commands retain every required resource.
- [ ] 11.4 Async track: deferred errors preserve operation context.
- [ ] 11.5 Allocation caching is measured and either bounded or rejected.
- [ ] 11.6 Selected execution model passes lifetime/reference stress.

## Exit Criteria

You are ready for Module 12 when:

1. The baseline and predeclared threshold justify the selected synchronous or
   async track.
2. Synchronization, error, and context-teardown points are public, documented,
   and tested for that track.
3. The selected execution model agrees with reference results and releases
   transient state across long chains.
4. On the async track, in-flight resources outlive GPU use and deferred errors
   retain actionable operation context.
5. On a rejected async experiment, the original synchronous semantics are
   restored and their complete test suite passes.
6. Any allocator cache exists only with allocation evidence and bounded tests.
