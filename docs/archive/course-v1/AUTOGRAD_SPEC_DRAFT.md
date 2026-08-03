# comtam Autograd Design

**Status:** Accepted design, not yet implemented.

This document defines the target ownership, runtime, recording, and reverse-mode autograd design for Module 6. It is intentionally separate from [`ARCHITECTURE.md`](../../ARCHITECTURE.md), which describes the source that exists today. When implementation and tests establish a decision here, move the implemented facts into `ARCHITECTURE.md` and keep this document as the design rationale.

The design preserves comtam's current boundaries: C++20, eager execution, Metal-only storage and kernels, float32 first, and one Apple GPU device per runtime. It does not introduce lazy execution, a compiler graph, or a global autograd tape.

## Accepted Decisions

| Area | Decision |
| --- | --- |
| Tensor identity | `tensor` becomes a cheap handle over `std::shared_ptr<TensorImpl>` |
| View results | Public movement creates a new `TensorImpl` while sharing storage and runtime |
| Runtime | Each `TensorImpl` owns a shared runtime identity; a process-wide default is only a convenience facade |
| Runtime isolation | An operation rejects operands from different runtimes before allocation or dispatch |
| Recording mode | `no_grad` is lexical, nestable, thread-local, and scoped to one runtime |
| Graph placement | Each differentiable result owns its producer node; there is no central tape |
| Backward rules | Typed subclasses of a polymorphic `grad_fn` base |
| Saved values | Detached dtype/view/storage snapshots, never saved autograd identities |
| Mutation detection | Saved storage versions are checked before a backward rule reads them |
| Intermediate gradients | Stored in a backward-local map, not persistently on non-leaf tensors |
| Persistent gradients | Only leaves own persistent gradient slots in Module 6 |
| Failure | Backward prepares results locally, commits leaf gradients only after success, then consumes the graph |
| Higher-order gradients | Deferred; backward formulas execute with recording disabled |

These decisions deliberately replace Module 6's earlier proposal to pass an explicit `core::context&` through every movement operation. Runtime identity is instead carried by the tensor handle, and a documented default runtime keeps the common C++ and future Python APIs concise.

## Goals And Non-Goals

The design must make these claims precise and testable:

- Copying a tensor handle preserves one logical tensor and one leaf gradient slot.
- A public movement result has a new logical identity even when it aliases the same Metal buffer.
- Every eager differentiable primitive records one semantic backward node when recording is enabled and at least one input requires gradients.
- Internal broadcast aliases and backward-support kernels never become graph nodes.
- A dynamic DAG is traversed once per logical tensor identity, while gradient contributions from every edge are added.
- Destroying outputs or consuming a graph releases graph-owned tensors and saved Metal buffers without a strong-reference cycle.
- The default runtime does not prevent explicit, isolated runtimes in tests or embedding applications.

Module 6 does not require retain-graph behavior, higher-order gradients, multithreaded backward, user-defined Python backward functions, distributed execution, or multiple devices inside one runtime. The polymorphic rule boundary may support future native rules, but that extensibility does not earn those features now.

## Terminology

- A **runtime** is the lifetime and identity shared by a Metal device, command queue, kernel library, and runtime-scoped framework services.
- The **default runtime** is the process-wide runtime selected when construction does not name another runtime.
- A **tensor handle** is the public `tensor` value copied by user code.
- A **TensorImpl** is the shared object that defines one logical tensor's identity, metadata, storage interpretation, runtime, and autograd metadata.
- A **leaf** is a user-created `TensorImpl` requiring gradients and having no producer.
- A **produced tensor** is the differentiable result of a recorded semantic operation.
- A **grad_fn instance** is the output-owned backward node for one recorded operation.
- A **saved tensor** is a detached snapshot of the value description needed by a backward rule; it has no autograd identity or producer.

## Target Architecture

```text
public tensor handle
  shares one TensorImpl
    owns runtime identity
    owns dtype and logical view
    shares Metal storage
    owns AutogradMeta
      classifies constant / leaf / produced / detached
      owns a persistent gradient only when it is a leaf
      owns one producer grad_fn when it is an available produced tensor

producer grad_fn
  strongly owns parent TensorImpl handles
  owns typed saved metadata or detached saved tensors
  never strongly owns its output TensorImpl

runtime state
  owns Metal device and queue
  owns kernel library
  does not own tensors or graph nodes
```

The strong ownership direction always points from a result toward its inputs:

```text
output TensorImpl -> producer grad_fn -> parent TensorImpl -> earlier producer
```

No producer owns its output, so this backward chain is acyclic by construction. Shared subgraphs form a DAG because several later nodes may own the same parent `TensorImpl`.

## Runtime Model

### Context As A Runtime Handle

The current `core::context` owns a device and kernel library directly. The target design keeps `core::context` as the public runtime handle but moves the owned resources behind shared runtime state:

```cpp
namespace comtam::core {

class context {
   public:
    context();
    context(const context&) = default;
    context& operator=(const context&) = default;

   private:
    std::shared_ptr<runtime_state> state_;
};

context& default_context();

}  // namespace comtam::core
```

The exact private layout is an implementation decision, but identity is the `runtime_state`, not the address of a temporary `context` facade. A tensor owns a shared handle to that state, so its Metal resources remain valid for the tensor's lifetime.

Copying a `context` facade shares one runtime identity. Independently default-constructed contexts create isolated runtime identities. `default_context()` is initialized lazily, is process-wide, and is not replaceable during Module 6; making default selection mutable would add synchronization and lifetime semantics that this module does not need.

The runtime must not retain tensors or graph nodes. This prevents the default runtime's process lifetime from turning every graph into process-lifetime state.

### Default And Explicit Construction

The common API uses the default runtime:

```cpp
auto x = tensor::from_vector(data, {2, 3}, /* requires_grad = */ true);
auto y = x.reshape({3, 2});
auto loss = tensor::sum(tensor::mul(y, y));
loss.backward();
```

Tests and embedding applications can select an explicit runtime:

```cpp
core::context runtime;
auto x = tensor::from_vector(data, {2, 3}, runtime, true);
```

Operations infer runtime identity from their tensor operands:

- A unary or movement operation inherits its input's runtime.
- A binary operation requires both inputs to have the same runtime.
- A scalar constant created by a composition is created in the non-constant operand's runtime.
- A produced tensor inherits the validated input runtime.
- `backward()` uses the scalar root's runtime.

An explicit-runtime overload may remain where it improves construction or testing, but ordinary tensor algebra does not require passing a context through every call.

### Cross-Runtime Rejection

Runtime mismatch is a metadata error and must fail before output allocation or Metal dispatch:

```text
add(tensor from runtime A, tensor from runtime B) -> error
```

Automatic copies between runtimes are not part of Module 6. A future transfer operation must be public and explicit because it creates new storage and a new runtime identity.

### Recording Mode

The default runtime and recording mode have different scopes:

```text
default runtime selection     process-wide convenience
recording suppression         thread-local and per runtime
autograd graph                owned by tensor results
```

`no_grad` uses a nesting depth rather than one Boolean. A guard targets either the default runtime or an explicit runtime:

```cpp
{
    no_grad guard;
    auto y = tensor::mul(x, x);  // forward executes; no producer is attached
}

{
    no_grad guard(runtime);
    auto y = tensor::mul(x, x);
}
```

Nested guards restore the previous depth on every exit path, including exceptions. Suppression in one thread does not affect another thread using the same runtime. Suppression for runtime A does not affect runtime B in the same thread.

The thread-local mechanism records only mode, not graph edges. There is no thread-local or process-global tape. A practical implementation may key a thread-local depth map by runtime-state identity while the guard retains that identity for its lexical lifetime.

Module 6 still excludes multithreaded backward. A small recording-mode test may use two threads to prove isolation without executing backward concurrently.

## TensorImpl And Tensor Identity

### Target Shape

The public tensor becomes a handle:

```cpp
class tensor {
   public:
    tensor(const tensor&) = default;
    tensor& operator=(const tensor&) = default;

   private:
    explicit tensor(std::shared_ptr<TensorImpl> impl);
    std::shared_ptr<TensorImpl> impl_;
};
```

`TensorImpl` groups all properties that must agree for one logical identity:

```cpp
struct TensorImpl {
    std::shared_ptr<core::runtime_state> runtime;
    DType dtype;
    view logical_view;
    std::shared_ptr<core::storage> storage;
    AutogradMeta autograd;
};
```

These sketches define responsibilities, not required member spelling or file layout.

### Copy, Movement, And Detach

```text
tensor b = a
  -> b.impl_ == a.impl_
  -> same identity, view, storage interpretation, producer, and leaf grad slot

auto b = a.reshape(...)
  -> b.impl_ != a.impl_
  -> same runtime and storage
  -> new view and new autograd identity
  -> reshape producer when recording applies

auto b = a.detach()
  -> b.impl_ != a.impl_
  -> same runtime, dtype, view, and storage
  -> detached classification, requires_grad=false, no producer
```

Raw aliases used by kernel submission do not construct a public `tensor` or `TensorImpl`. They remain local dtype/view/storage descriptors. The raw broadcast aliases already used by binary dispatch are the intended pattern.

### Autograd Classification

Independent booleans can encode contradictory states, so classification is explicit:

```cpp
enum class autograd_kind {
    constant,
    leaf,
    produced,
    detached,
};

enum class backward_state {
    available,
    consumed,
};
```

The valid states are:

| Kind | Requires gradients | Producer | Persistent gradient |
| --- | --- | --- | --- |
| `constant` | No | None | None |
| `detached` | No | None | None |
| `leaf` | Yes | None | Optional leaf slot |
| `produced`, available | Yes | Required | None |
| `produced`, consumed | Yes | Released | None |

A scalar leaf may be a backward root. It receives the seed gradient directly. A consumed produced tensor must never be reclassified as a leaf merely because its producer has been released.

## Backward Node Model

### Typed Runtime Polymorphism

The graph is heterogeneous at runtime, so Module 6 uses a polymorphic backward node with typed final subclasses:

```cpp
class grad_fn {
   public:
    virtual gradient_list apply(const tensor& upstream,
                                autograd_engine& engine) = 0;
    virtual std::span<const tensor_handle> parents() const noexcept = 0;
    virtual void validate_saved_values() const = 0;
    virtual void release_saved_values() noexcept = 0;
    virtual std::string_view name() const noexcept = 0;
    virtual ~grad_fn() = default;
};
```

For example, a multiplication node owns two parent handles and the detached values required by its local derivative:

```cpp
class mul_backward final : public grad_fn {
    std::array<tensor_handle, 2> parents_;
    saved_tensor lhs_;
    saved_tensor rhs_;
};
```

The base interface is the graph boundary. It must not absorb forward Metal dispatch, public operator validation, or device scheduling. Forward primitives remain direct eager tensor operations; after successful execution they attach one typed node when recording applies.

This is intentionally harder than a central `std::variant`, but the additional mechanism is relevant: it teaches heterogeneous graph ownership, typed saved state, runtime dispatch, and explicit release. It also permits future native rules without editing a central variant. Custom Python backward functions remain deferred.

### Parent Ownership

A node strongly owns every differentiable parent needed for traversal. It may omit a constant parent from graph traversal, but a rule must still save any constant value needed for its derivative.

Raw pointers into caller-local tensor variables are forbidden. Parent identity is a shared `TensorImpl` handle, and traversal compares that stable identity, not tensor values or overloaded equality.

### One Node Per Semantic Primitive

Forward recording follows the semantic operation surface:

```text
add, mul, neg, recip, sum, matmul, public movement -> typed nodes
sub, div, mean                                  -> primitive compositions
max, min                                        -> reject differentiable recording
internal broadcast aliases                     -> no node
internal zero_pad used by backward              -> no node
physical kernel variants                        -> no node
```

Composed operations validate their complete public contract before executing the first internal primitive. Their primitive calls then record exactly the same nodes as the equivalent user-written expression.

## Saved Values And Mutation

### Detached Saved Tensor

A backward rule must never save its output as a normal tensor handle. Doing so would create this cycle:

```text
output TensorImpl -> producer grad_fn -> output TensorImpl
```

Instead, rules save only a detached value description:

```cpp
struct saved_tensor {
    std::shared_ptr<core::runtime_state> runtime;
    DType dtype;
    view logical_view;
    std::shared_ptr<core::storage> storage;
    std::uint64_t expected_storage_version;
};
```

This retains the Metal bytes and their interpretation without retaining an autograd identity or earlier producer chain. `recip` may therefore save its forward output for `-dout*y*y` without forming a cycle.

### Storage Versions

`core::storage` gains a monotonically increasing version. Every public write to existing storage increments it, including host writes and future optimizer updates. Allocation and initialization establish the initial version.

Before reading a saved tensor, a rule compares the current and expected versions. A mismatch fails backward with the operation name and saved/current versions. Silently differentiating changed values is forbidden.

Module 6 does not need in-place arithmetic, but `from_vector` is already a mutation path and must participate in versioning. Parameter updates in Module 7 will reuse the same rule.

## Forward Recording Algorithm

Every differentiable semantic primitive follows this sequence:

```text
1. Infer and validate one runtime from the operands.
2. Validate dtype, shape, view, and operation-specific metadata.
3. Execute the eager Metal forward operation.
4. Construct the result TensorImpl in the inferred runtime.
5. Inspect thread-local recording mode for that runtime.
6. If recording is enabled and any input requires gradients:
     classify the result as produced,
     construct the typed grad_fn,
     attach it as the result's producer.
   Otherwise:
     classify the result as constant.
```

Forward failure never creates a partial graph node. Node attachment occurs only after successful eager execution.

Forward-only operations such as `max` may execute on constants. If recording is enabled and a differentiable input reaches an operation without an accepted rule, the operation fails clearly rather than silently detaching the result.

## Backward Engine

### Responsibilities

The backward engine is an execution helper, not a graph owner. One invocation owns:

- the topological order;
- the visited identity set;
- temporary upstream gradients;
- prepared replacement values for leaf gradient slots;
- the scoped recording suppression used by local rules.

No invocation state survives successful or failed backward.

### Validation

Before traversal, `backward()` rejects:

- a non-scalar root;
- a constant or detached root;
- a produced root whose graph is consumed;
- a reachable consumed produced tensor;
- a reachable saved tensor whose storage version changed;
- a runtime mismatch inside the reachable graph.

The validation and traversal identify tensors by `TensorImpl` address while holding shared handles for the invocation's lifetime.

### Topological Traversal

Traversal starts at the root and follows producer parents. A visited set is required because the graph is a DAG. A node is appended after all reachable parents, producing parent-before-child topological order. Backward processes the order in reverse.

Recursive traversal is acceptable for the first implementation only if the supported graph-depth limit is explicit. An iterative traversal avoids C++ call stack exhaustion and is the preferred robust form.

### Local Gradient Map

Intermediate gradients live in an invocation-local map keyed by `TensorImpl` identity:

```text
TensorImpl* -> detached tensor gradient
```

The root receives a rank-0 value of one in its runtime. Every returned parent contribution is shape-checked before insertion. Multiple contributions use the existing eager `add` primitive while recording is suppressed.

Non-leaf gradients are erased when they are no longer needed. They never enter persistent tensor state in Module 6.

### Transactional Leaf Commit

Backward formulas and accumulation can fail because they execute real eager operations. Persistent leaf gradients must not be partially updated.

The engine therefore prepares each final leaf value locally. If a leaf already has a gradient, the engine eagerly computes `old_grad + contribution` under recording suppression before committing anything. Only after every rule and preparation succeeds does it replace the affected leaf gradient handles.

The final commit consists only of non-throwing handle assignments. This is the transaction boundary:

```text
failure before commit -> leaf gradients and graph remain unchanged
successful commit     -> leaf gradients update, then graph is consumed
```

### Consumption

After a successful commit, every visited produced `TensorImpl` becomes consumed, releases its producer, and therefore releases producer-owned parents and saved values when no other owner remains. A second backward through any consumed part of that graph fails clearly.

If backward fails before commit, the graph remains available for inspection or retry and no persistent leaf gradient changes. This failure behavior is an accepted part of the design.

## Representative Trace: `z = x*x + x`

Assume `x` is a scalar leaf with `requires_grad=true`.

### Forward

```text
x TensorImpl (leaf)

m = mul(x, x)
m TensorImpl (produced)
└── mul_backward
    ├── parent x
    ├── parent x
    ├── saved x value for lhs derivative
    └── saved x value for rhs derivative

z = add(m, x)
z TensorImpl (produced)
└── add_backward
    ├── parent m
    └── parent x
```

The repeated parent in `mul_backward` represents two graph edges but one tensor identity.

### Topological Order

Identity-based traversal visits `x` once:

```text
[x, m, z]
```

Backward processes:

```text
[z, m, x]
```

### Gradient Accumulation

```text
grad[z] = 1

add_backward(z):
  grad[m] += 1
  grad[x] += 1

mul_backward(m):
  first x edge:  grad[x] += 1 * x
  second x edge: grad[x] += 1 * x

prepared leaf gradient:
  grad[x] = 1 + x + x = 2*x + 1
```

The engine processes the `mul` node once, preserves both parent edges, and accumulates three contributions into one leaf identity.

### Successful Completion

```text
x.grad <- prepared 2*x + 1
m      -> consumed, producer released
z      -> consumed, producer released
```

Destroying `z` after backward does not affect `x.grad`. Destroying `z` before backward releases the `add` node and, when otherwise unreachable, the `mul` node and its saved storage.

## Public API Direction

The exact spelling will be implemented incrementally, but the ownership model supports this target:

```cpp
auto x = tensor::from_vector(values, shape, /* requires_grad = */ true);
auto y = x.reshape(new_shape);
auto z = tensor::add(tensor::mul(y, y), y);
auto loss = tensor::sum(z);

loss.backward();
auto dx = x.grad();
x.clear_grad();

{
    no_grad guard;
    auto detached_update = tensor::sub(x, step);
}
```

Explicit runtime construction remains available:

```cpp
core::context runtime;
auto x = tensor::from_vector(values, shape, runtime, true);

{
    no_grad guard(runtime);
    // Operations on tensors from runtime remain non-recording in this scope.
}
```

Operator overloads are optional API sugar and should be added only after the named functions and ownership semantics are tested.

## Error And Lifetime Invariants

The implementation and tests must preserve these invariants:

1. A public tensor handle always owns exactly one non-null `TensorImpl`.
2. One `TensorImpl` belongs to exactly one runtime identity.
3. A binary operation never mixes runtime identities implicitly.
4. A produced, available differentiable tensor has exactly one producer.
5. A producer strongly owns parents and never strongly owns its output.
6. A saved tensor owns value resources but no autograd identity.
7. Only leaves have persistent gradient slots.
8. Temporary gradients belong to one backward invocation.
9. Recording is disabled for every backward formula and accumulation op.
10. Successful backward commits all affected leaf gradients before consuming any graph node.
11. Failed backward commits no leaf gradient and consumes no graph node.
12. A consumed produced tensor never becomes indistinguishable from a leaf.
13. Default-runtime lifetime does not retain tensors or graphs.
14. `no_grad` nesting, thread isolation, and runtime isolation are observable and tested.

## Required Design Tests

The implementation plan must include tests for:

- tensor copies sharing one `TensorImpl` and gradient slot;
- public movement producing a distinct identity while sharing runtime/storage;
- detach producing a distinct non-recording identity;
- explicit and default runtime construction;
- cross-runtime binary rejection before dispatch;
- nested `no_grad` restoration after normal and exceptional exit;
- `no_grad` isolation across runtimes and threads;
- temporary parent lifetime through backward;
- no output-to-node-to-output strong cycle;
- repeated parent edges and diamond graphs;
- backward failure leaving leaf gradients and graph state unchanged;
- successful backward consuming and releasing reachable graph state;
- saved-storage mutation rejection;
- framework-owned live node counts returning to baseline in repeated steps.

Every mathematical rule additionally requires the numerical and exact-shape tests specified by Module 6.

## Rejected Alternatives

### One Process-Wide Singleton As The Only Runtime

This gives a short API but prevents isolated tests, makes runtime mismatch undetectable, couples recording state across unrelated work, and makes future Python threading semantics difficult to repair. A global default plus tensor-bound runtime identity provides the same common-case API without those costs.

### Explicit Context On Every Tensor Operation

This is simple and remains valid for construction, but it makes public movement and future Python bindings carry runtime arguments that are already determined by their operands. Tensor-bound runtime identity removes that duplication and makes mismatches checkable.

### Central Global Tape

A central tape extends graph lifetime beyond output lifetime and requires manual clearing and synchronization. Output-owned nodes let ordinary shared ownership express graph reachability.

### Separate Shared Autograd State Beside Value Metadata

Copying dtype/view/storage by value while sharing only autograd state allows identity and metadata to diverge. Putting both in `TensorImpl` makes the unit of identity explicit.

### Backward Lambdas Capturing Tensors

`std::function` closures hide captures, saved storage, and cycle risks. Typed `grad_fn` subclasses expose ownership and release behavior.

### Persistent `.grad` On Every Tensor

This retains non-leaf temporaries and complicates partial failure cleanup. A backward-local map is sufficient until retained non-leaf gradients are an explicit feature.

## Implementation Sequence

This document does not authorize broad one-shot implementation. Preserve independently reviewable checkpoints:

1. Introduce shared runtime state and `TensorImpl` handle identity without autograd rules; preserve all forward tests.
2. Add default/explicit construction, runtime inference, and cross-runtime rejection.
3. Add runtime-specific thread-local recording mode, tensor classification, detach, and identity/lifetime tests.
4. Add the `grad_fn` base, one representative `mul`/`add` graph, traversal, transactional accumulation, and consumption.
5. Add the numerical gradient checker and primitive rules in dependency order.
6. Add reduction, broadcast, movement, matmul, and internal `zero_pad` rules.
7. Add storage versioning, mutation rejection, and repeated-step lifetime evidence.

Each checkpoint must keep forward correctness intact and must not claim a Module 6 gate beyond the tests actually executed.

## Deferred And Open Decisions

The accepted ownership model does not require deciding these details yet:

- exact header/source placement and private type spelling;
- whether `requires_grad` is construction-only or has a constrained leaf setter;
- whether `grad()` returns a tensor handle, optional handle, or checked handle;
- operator-overload syntax;
- iterative traversal container details;
- public debugging hooks beyond deterministic test-only live counts;
- retained non-leaf gradients, retain-graph, and higher-order recording;
- custom native or Python backward registration;
- transfer semantics between runtimes.

Resolve each when its implementation checkpoint begins. None may weaken the accepted ownership, runtime isolation, transaction, or graph-consumption invariants silently.
