# Autograd design

This document records the accepted direction for adding dynamic reverse-mode
autograd to `comtam`. Implemented runtime and identity contracts are separated
from planned autograd behavior so contributors do not mistake a design for a
working feature.

## Implemented contracts

- **Tensor identity.** `tensor` is a handle over
  `std::shared_ptr<tensor_impl>`.
- **View results.** Movement creates a new `tensor_impl`. The result shares
  storage and runtime with its input.
- **Runtime.** Every `tensor_impl` owns shared runtime identity.
- **Runtime isolation.** Mixed-runtime operations reject before allocation or
  dispatch.

## Planned contracts

- **Gradient requirement.** Store `requires_grad` on `tensor_impl`.
- **Recording suppression.** Make `no_grad` lexical, nestable, thread-local,
  and runtime-specific.
- **Detach.** Create a new implementation over shared value state without
  gradient recording.
- **Graph placement.** Let differentiable results own producer nodes. Do not
  use a central tape.
- **Rule representation.** Use typed subclasses of a polymorphic `grad_fn`
  base.
- **Saved values.** Save detached runtime, dtype, view, and storage snapshots.
  Include storage versions.
- **Accumulation.** Key backward-local gradients by `tensor_impl` identity.
- **Persistent gradients.** Retain gradient slots only on leaves initially.
- **Failure behavior.** Prepare locally. Commit leaf gradients after success,
  then consume the graph.
- **Higher-order gradients.** Defer them. Execute backward formulas without
  recording.

## Constraints

- Copying a tensor preserves logical identity.
- Public movement creates a distinct logical identity even when storage is
  shared.
- Backward ownership must not create a cycle between an output and its producer
  node.
- Gradient rules require independent numerical checks.
- Recording through `max` remains unsupported until the tie policy is explicit
  and tested.

## Changing the design

A design change must state the disputed assumption, its concrete consequence,
the replacement mechanism, and the tests that distinguish the alternatives.
Update this document with the implementation so the contract does not drift.
