# Autograd Track Decision Reference

**Status:** Accepted direction, explained incrementally by the tutorial chapters.

This file records the choices that chapters must explain and implementations must not silently contradict. It is intentionally compact. If a row is difficult to understand, return to the corresponding chapter rather than treating the row as sufficient teaching material.

| Area | Accepted direction | Teaching chapter |
| --- | --- | --- |
| Tensor identity | `tensor` becomes a cheap handle over `std::shared_ptr<tensor_impl>` | Chapter 2 |
| View results | Public movement creates a new `tensor_impl` while sharing storage and runtime | Chapter 2 |
| Runtime | Every `tensor_impl` owns shared runtime identity; construction may use a process-wide default | Chapter 1-2 |
| Runtime isolation | Mixed-runtime operations reject before allocation or dispatch | Chapter 1-2 |
| Constructor order | Tensor constructors use value/shape, then optional dtype, then optional context; selecting a context therefore spells the dtype | Chapter 2 |
| Recording | `no_grad` is lexical, nestable, thread-local, and runtime-specific | Chapter 3 |
| Graph placement | Differentiable results own producer nodes; there is no central tape | Chapter 4 |
| Rule representation | Typed subclasses of a polymorphic `grad_fn` base | Chapter 4 |
| Saved values | Detached runtime/dtype/view/storage snapshots with storage versions | Chapter 4 and 6 |
| Intermediate gradients | A backward-local map is keyed by `tensor_impl` identity | Chapter 5 |
| Persistent gradients | Only leaves retain gradient slots in the first autograd system | Chapter 5 |
| Backward failure | Prepare locally, commit leaves after success, then consume the graph | Chapter 5 |
| Higher-order gradients | Deferred; backward formulas execute without recording | Chapter 3 and 5 |

## Changing A Decision

A decision may change when implementation evidence shows a better design, but the change must state the disputed assumption, the concrete consequence, the replacement mechanism, and the tests that distinguish the designs. Update the relevant chapter and this table together.
