# Chapter 2: Tensor Identity And `tensor_impl`

**Status:** Implemented. Gate met.

Chapter 1 gave every tensor a runtime. This chapter answers the question that raises: what object *is* a tensor, once "the same tensor" has to mean something to a gradient?

## The Concrete Problem

Two expressions that look almost identical must behave differently:

```cpp
tensor b = a;                 // b and a are the same logical tensor
auto  c = a.reshape({2, 6});  // c is a different logical tensor, over the same bytes
```

If `a` requires a gradient, `b` must accumulate into `a`'s single gradient slot - every edge, one slot. But `c` must own its own producer node, because backward has to undo the reshape before contributing anything to `a`. Sharing `a`'s full autograd identity would erase that edge.

So the requirement exists before any derivative formula does:

```text
ordinary handle copy -> share identity
semantic operation   -> create identity
```

## Four Identities You Must Not Confuse

Autograd discussions go vague when "identity" is used without naming the layer. comtam needs four separate concepts, and the design is only correct if each stays distinct.

**C++ handle identity.** `tensor b = a;` makes two C++ objects at different addresses. That fact says nothing about whether they are the same logical tensor.

**Logical tensor identity.** Where `requires_grad`, a producer node, and a leaf gradient slot belong. In comtam this is the shared `tensor_impl` object: `a.impl_ == b.impl_`.

**Storage identity.** Whether two tensors refer to the same Metal allocation. Different logical tensors routinely share storage - every movement operation is an example.

**Value equality.** Whether two tensors currently hold equal values. Two independent allocations can be value-equal with different identity at every other layer.

Never substitute value equality, metadata equality, or storage equality for logical identity.

## The Semantic Matrix

The behavior the representation has to express:

| Expression | Same logical identity? | Same storage? | Same runtime? | Same view? |
| --- | --- | --- | --- | --- |
| `tensor b = a` | Yes | Yes | Yes | Yes |
| `auto b = a.reshape(...)` | No | Yes | Yes | Usually no |
| `auto b = a.transpose(...)` | No | Yes | Yes | No |
| `auto b = a.detach()` | No | Yes | Yes | Yes |
| `auto b = tensor::neg(a)` | No | No | Yes | Result view |
| independently construct equal values | No | No | Maybe | Maybe |

A design is right only when every row falls out of ownership naturally. A design that needs a special-case identity flag per row is fighting its own object model.

The `detach()` row is the sharpest one, even though `detach` itself belongs to Chapter 3: equal runtime, dtype, view, *and* storage still must not imply equal logical identity.

## Designs Considered

**The address of the public `tensor`.** Fails immediately - ordinary copies have different addresses, and temporaries vanish while graph nodes still need their inputs. A graph holding raw pointers to caller-local handles would depend on variables staying alive by accident.

**`storage*` as identity.** Copies work, because they share storage. But it merges every view of one allocation into a single identity:

```text
x                 -+
x.transpose(...)   |-- same storage, different logical tensors
x.shrink(...)      |
x.expand(...)     -+
```

Movement results could then never own distinct producer nodes.

**A separate shared `autograd_meta`, with dtype/view/storage still on the handle.** Workable, but it splits one logical identity across independently copied parts. Nothing structurally prevents one header copy from changing while both copies still claim the same gradient identity. The invariant "all handles sharing `autograd_` must have identical runtime/dtype/view/storage interpretation" would have to be maintained by hand, in every operation, forever.

**One shared `tensor_impl`.** Runtime, dtype, view, storage interpretation, and later autograd metadata live together in one shared object ([impl.h:25](../../../comtam/tensor/impl.h:25)). Copying a tensor copies one pointer ([tensor.h:279](../../../comtam/tensor/tensor.h:279)); a semantic operation constructs another implementation.

```text
same impl_      -> same complete logical tensor identity
different impl_ -> different logical tensor identity
```

This is the chosen design, because the ownership graph itself enforces the semantic distinction instead of a comment asking future code to be careful.

## How The Matrix Falls Out

**Copy** is structural, not a policy: `tensor`'s copy constructor is `= default` ([tensor.h:30](../../../comtam/tensor/tensor.h:30)), so it copies one `shared_ptr`. `is_same` ([tensor.h:34](../../../comtam/tensor/tensor.h:34)) makes the contract executable by comparing `impl_` directly.

**Movement** builds a new view over the same storage and wraps it in a *new* implementation - see `permute`, `transpose`, `shrink`, `expand`, `reshape` ([tensor.h:157-190](../../../comtam/tensor/tensor.h:157)). All five follow the same three-line shape: derive the view, `make_impl_from_storage` with the existing storage and runtime, return a tensor over the new impl. Note that no movement returns `*this`, even when the operation happens to be a no-op: a caller that writes `a.reshape(a.shape())` still gets a distinct identity, because otherwise identity would depend on runtime values.

**Allocating operations** create a new implementation *and* new storage in the operand runtime ([tensor.cpp:20](../../../comtam/tensor/tensor.cpp:20) and the other three dispatch functions below it).

**Mixed runtimes** are rejected before allocation or dispatch. `check_same_runtime` ([checks.h:49](../../../comtam/utils/checks.h:49)) compares the shared runtime pointers, and binary dispatch calls it first thing.

## Raw Internal Aliases Are Not Tensor Identities

Broadcasting needs expanded views of both operands. The tempting implementation builds temporary tensors for them. Once every constructed tensor carries an identity, that would mint a logical tensor for something that exists only to describe kernel indexing - and later recording could mistake it for a public `expand` performed by the user.

Binary dispatch keeps them as raw `view` values instead ([tensor.cpp:35](../../../comtam/tensor/tensor.cpp:35)), passed straight into the command descriptor. They own no storage, no runtime, and no autograd state; the original operand implementations stay the semantic inputs.

This is not a micro-optimization. It determines the shape of the future graph:

```text
correct:   add output -> add_backward -> original a, original b
incorrect: add output -> hidden expand nodes -> temporary alias identities
```

## Constructor Order Is An API Decision

Tensor constructors previously took `core::metal_device&`. A device reference cannot supply the shared runtime state a tensor must retain, and there is no safe way to recover the owning `runtime_state` from an arbitrary device reference - so "add tensor-bound runtime without touching constructors" was never available.

The accepted order is value/shape, then dtype, then context ([tensor.h:44](../../../comtam/tensor/tensor.h:44)):

```cpp
tensor(data, shape);                          // default dtype and runtime
tensor(data, shape, DType::Float32);          // explicit dtype, default runtime
tensor(data, shape, DType::Float32, context); // explicit dtype and runtime
```

Trailing `context` preserves the established `tensor(shape, dtype)` form and keeps the rarely-changed argument last. The deliberate cost: selecting a non-default context also means spelling the dtype. That is preferable to inserting runtime selection between shape and dtype, or to maintaining a larger overload family. The same order applies to scalar, shape, and view construction - there is no remaining path that creates a tensor without runtime identity.

## Ownership Traces

Worth being able to draw from memory:

```text
ordinary copy         a.impl_ ── shared ──> impl ──> storage, runtime
                      b.impl_ ─────────────┘

public movement       a.impl_ ──> impl A ──┐
                      b.impl_ ──> impl B ──┴──> same storage, same runtime

allocating op         a.impl_ ──> impl A ──> storage A ─┐
                                                        ├── same runtime
                      out.impl_ ─> impl OUT ─> storage OUT ┘

destruction           last tensor_impl dies -> storage refcount drops -> buffer freed
                                            -> runtime refcount drops -> device freed last
```

The destruction order matters and is enforced by member declaration order in `tensor_impl` ([impl.h:39](../../../comtam/tensor/impl.h:39)): `runtime_state` is declared before `storage`, so storage is destroyed first and the device outlives the buffers it allocated.

## Where Autograd Metadata Will Go

`tensor_impl` is now the obvious home for `requires_grad`, a producer node, and a leaf gradient slot - all three describe the logical tensor, not the handle and not the bytes. None of them were added during this chapter. Mixing an identity refactor with recording state would have made identity bugs and recording bugs indistinguishable. Chapter 3 adds the first of the three.

## Common Mistakes

**Using storage as identity.** Merges every view of one allocation.

**Deep-copying `tensor_impl` in the copy constructor.** Silently breaks gradient accumulation: each handle gets its own slot and neither sees the sum.

**Keeping legacy dtype/view/storage fields beside `impl_`.** Two sources of truth; one of them will go stale.

**Returning `*this` from a no-op movement.** Makes identity depend on runtime values.

**Turning internal broadcast aliases into tensors.** Injects phantom nodes into the future graph.

**Recovering runtime from a `metal_device&`.** Not possible safely; every construction path must be handed a runtime.

## Chapter Gate

- Every public tensor handle owns exactly one non-null shared `tensor_impl`.
- Copy construction and copy assignment share one logical identity.
- Every public movement creates a new identity while sharing storage and runtime.
- Every allocating operation creates a new identity and storage in the operand runtime.
- `tensor_impl` owns runtime, dtype, logical view, and storage interpretation together.
- No legacy dtype/view/storage fields remain on the public handle.
- No construction path lacks runtime identity.
- Mixed-runtime operands reject before allocation or dispatch.
- Internal broadcast aliases stay raw metadata.
- Context destruction cannot invalidate surviving tensors.
- No recording state, producer node, or gradient slot exists yet.

## Agent Feedback / Grading

**Status: passed.**

`tensor` has one non-null shared implementation identity, preserved under copy construction and assignment. Public movement creates distinct implementations sharing storage and runtime. Allocating unary, binary, reduction, and matmul operations create fresh storage in the operand runtime. Mixed-runtime binary operations reject before dispatch, composed `mean` keeps its scale constant in the operand runtime, and binary broadcasting uses raw temporary views rather than hidden tensor identities.

[`tests/tensor/identity.cpp`](../../../tests/tensor/identity.cpp) distinguishes handle, implementation, storage, runtime, and value identity; covers every public movement; checks allocating operations; exercises context-handle destruction; and rejects mixed runtimes across `add`, `sub`, `mul`, `div`, and `matmul`. The existing MLX-backed forward tests remain the independent numerical oracle.

One claim is worth stating precisely: "rejects before dispatch" is established jointly by the rejection tests and by inspection of validation order in [tensor.cpp:21](../../../comtam/tensor/tensor.cpp:21). The tests alone do not instrument submission counts.

Verification recorded at grading time:

```text
cmake --build build --target comtam_tests -j 6  -> compiled
ctest --test-dir build --output-on-failure      -> passed, 51/51 outside the sandbox
```

## What Comes Next

[Chapter 3](03_RECORDING_AND_NO_GRAD.md) adds the first piece of autograd state to `tensor_impl` and answers when an eager operation should attach a node at all - including how to turn recording off for a scope without inventing hidden global state.
