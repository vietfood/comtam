# Chapter 2: Tensor Identity And `tensor_impl`

**Status:** Implemented and independently verified; track advancement waits on Chapter 1's direct context tests.

[Previous: Runtime Ownership](01_RUNTIME_OWNERSHIP.md) | [Autograd track index](INDEX.md) | [Decision reference](DECISIONS.md)

Chapter 1 separated a small `core::context` handle from the shared runtime state that actually owns Metal execution resources. Chapter 2 answers the next ownership question:

> If `tensor b = a` must preserve one gradient identity, but `auto b = a.reshape(...)` must create a new identity while sharing storage, which C++ object represents the logical tensor?

The answer is an internal shared object named `tensor_impl`. PyTorch uses the spelling `TensorImpl`; comtam keeps its existing lowercase `snake_case` type convention. The idea matters more than the spelling: public `tensor` values become cheap handles, while `tensor_impl` becomes the object that defines one logical tensor identity.

Do not add gradient fields in this chapter. First make identity, runtime, view, and storage ownership correct and prove that every existing forward operation preserves its behavior. Autograd metadata will have a trustworthy home only after this chapter passes.

## 2.1 What The Current `tensor` Means

The current class directly stores three fields:

```cpp
class tensor {
    DType dtype_;
    view view_;
    std::shared_ptr<core::storage> storage_;
};
```

This is a value-type header over shared Metal bytes. Consider an ordinary copy:

```cpp
tensor b = a;
```

C++ copies `dtype_` and `view_` by value and copies `storage_` as a `shared_ptr`:

```text
a tensor object                    b tensor object
├── dtype_: Float32                ├── dtype_: Float32       copied value
├── view_: shape/strides/offset    ├── view_: same metadata copied value
└── storage_ ──────────────────────┴── storage_              shared owner
```

Today this works because dtype and view metadata are effectively immutable after construction, while shared storage intentionally makes writes visible through aliases. The runtime has not needed a separate logical identity yet.

Movement operations also construct another tensor header over the same storage:

```cpp
tensor tensor::reshape(const view_vector& shape) const {
    return tensor(storage_, view_.reshape(shape), dtype_);
}
```

An ordinary copy and a movement result therefore both produce another C++ `tensor` object, but autograd needs them to mean different things:

```text
tensor b = a             same logical tensor identity
auto c = a.reshape(...)  new logical tensor identity, shared storage
```

The current representation has no single object that expresses this distinction.

## 2.2 Four Identities You Must Not Confuse

Autograd discussions become vague when the word "identity" is used without naming the layer. Comtam needs four separate concepts.

### C++ Handle Identity

Two local variables are two C++ objects:

```cpp
tensor a = ...;
tensor b = a;

REQUIRE(&a != &b);
```

Their stack addresses differ. This fact says nothing about whether they represent the same logical tensor.

### Logical Tensor Identity

Logical identity answers where `requires_grad`, a producer node, and a leaf gradient slot belong. After this chapter, logical identity is the shared `tensor_impl` object.

```text
a.impl_ == b.impl_  -> same logical tensor
```

### Storage Identity

Storage identity answers whether tensors refer to the same Metal allocation.

```text
a.storage == c.storage  -> shared bytes
```

Different logical tensors may share storage. Every public movement operation is the important example.

### Value Equality

Value equality asks whether two tensors currently contain equal logical values. Two independently allocated tensors can be value-equal while having different tensor and storage identities.

Do not use value equality, metadata equality, or storage equality as a substitute for logical identity.

## 2.3 The Required Semantic Matrix

Before choosing a C++ representation, write down the behavior it must express:

| Expression | Same logical identity? | Same storage? | Same runtime? | Same view? |
| --- | --- | --- | --- | --- |
| `tensor b = a` | Yes | Yes | Yes | Yes |
| `auto b = a.reshape(...)` | No | Yes | Yes | Usually no |
| `auto b = a.transpose(...)` | No | Yes | Yes | No |
| `auto b = a.detach()` | No | Yes | Yes | Yes |
| `auto b = tensor::neg(a)` | No | No | Yes | Result view |
| independently construct equal values | No | No | Maybe | Maybe |

The representation is correct only if all rows are natural consequences of ownership. A design that needs special-case identity flags for every row is fighting its own object model.

`detach()` belongs to Chapter 3, but its row matters now: it proves that equal runtime, dtype, view, and storage still do not imply equal logical identity.

## 2.4 Why Autograd Forces An Explicit Identity

Suppose `x` is a leaf requiring gradients:

```cpp
tensor alias = x;
auto loss = tensor::mul(alias, alias);
loss.backward();
```

Both multiplication edges must accumulate into one leaf gradient slot because `alias` and `x` are the same logical tensor. If copying a tensor copied its gradient metadata, each handle could receive a separate partial gradient and neither would observe the correct sum.

Now consider a movement:

```cpp
auto transposed = x.transpose(0, 1);
```

The movement result needs its own producer node because backward must apply the inverse transpose before contributing to `x`. Sharing `x`'s complete autograd identity would erase that edge.

The identity requirement therefore exists before any derivative formula:

```text
ordinary handle copy -> share identity
semantic operation   -> create identity
```

## 2.5 Plausible Designs And Their Consequences

### Option A: Use The Address Of The Public `tensor`

```cpp
const tensor* identity = &tensor_object;
```

This fails immediately because ordinary copies have different addresses. Temporaries also disappear while graph nodes still need their inputs. A graph storing raw pointers to caller-local handles would depend on variables remaining alive accidentally.

### Option B: Use `storage*` As Tensor Identity

This makes copies work because they share storage, but it incorrectly merges every view:

```text
x                 ─┐
x.transpose(...)   ├── same storage, different logical tensors
x.shrink(...)      ┤
x.expand(...)     ─┘
```

If storage were identity, movement operations could not own distinct producer nodes or gradient mappings.

### Option C: Share Only A Separate `autograd_meta`

```cpp
class tensor {
    DType dtype_;
    view view_;
    std::shared_ptr<storage> storage_;
    std::shared_ptr<autograd_meta> autograd_;
};
```

This is workable if tensor metadata is permanently immutable and every operation carefully decides whether to reuse or replace `autograd_`. Its weakness is that one logical identity is split across independently copied dtype, view, storage, and autograd objects. A bug can change one header copy while both copies still claim the same gradient identity.

The representation would need to maintain this invariant manually:

```text
all tensor headers sharing autograd_ must always have identical runtime/dtype/view/storage interpretation
```

That is exactly the invariant an object should express structurally instead.

### Option D: Share One `tensor_impl`

```cpp
class tensor {
    std::shared_ptr<tensor_impl> impl_;
};
```

Runtime identity, dtype, view, storage interpretation, and later autograd metadata live together in the shared implementation. Copying a tensor copies one pointer. A semantic operation constructs another implementation.

```text
same impl_      -> same complete logical tensor identity
different impl_ -> different logical tensor identity
```

We choose Option D because the ownership graph itself enforces the semantic distinction.

## 2.6 Reading Your Current `tensor_impl` Draft

The current worktree contains this useful first draft:

```cpp
struct tensor_impl {
    DType dtype;
    view view;
    std::shared_ptr<core::storage> storage;
};
```

This is the correct inventory for an identity-only first checkpoint: dtype, view, and storage interpretation move out of the public handle together. Two details need attention before using it.

First, include `<memory>` directly because this header names `std::shared_ptr`. Relying on another included header to include `<memory>` is a fragile transitive dependency.

Second, avoid naming a member `view` when its type is also `view`. Even when accepted by a compiler, expressions such as `impl_->view` make type and object roles harder to scan. `logical_view` states what the member means and distinguishes it from raw internal kernel aliases.

An identity-only checkpoint can therefore begin with:

```cpp
#pragma once

#include <memory>
#include <stdexcept>
#include <utility>

#include "comtam/core/storage.h"
#include "comtam/macros/log.h"
#include "comtam/tensor/dtype.h"
#include "comtam/tensor/view.h"

namespace comtam {

struct tensor_impl {
    tensor_impl(DType dtype_value, view view_value,
                std::shared_ptr<core::storage> storage_value)
        : dtype(dtype_value),
          logical_view(std::move(view_value)),
          storage(std::move(storage_value)) {
        COMTAM_CHECK_AND_THROW(storage, std::runtime_error,
                               "tensor_impl requires storage");
    }

    DType dtype;
    view logical_view;
    std::shared_ptr<core::storage> storage;
};

}  // namespace comtam
```

The direct includes matter. This header names `std::shared_ptr`, `std::runtime_error`, `std::move`, and `COMTAM_CHECK_AND_THROW`, so it includes the standard and project headers that define those names instead of inheriting them accidentally through `storage.h`.

Do not add a default constructor. A default-constructed implementation with null storage would create an invalid logical tensor that every method must defend against.

## 2.7 Why Start With An Identity-Only Checkpoint?

The final `tensor_impl` must also own shared runtime identity, but adding handle identity and changing every constructor from `metal_device&` to runtime-aware construction in one edit creates too many failure sources.

Your current draft provides a useful intermediate checkpoint:

```text
Checkpoint A
  tensor becomes shared_ptr<tensor_impl>
  tensor_impl contains dtype/view/storage
  existing explicit-context operation API remains
  all forward tests must remain unchanged

Checkpoint B
  tensor_impl gains shared runtime_state
  construction chooses explicit or default context
  operations infer and validate runtime identity
```

Checkpoint A is not the final design, but it is internally honest: it makes no false runtime claim. Checkpoint B then adds a required non-null runtime rather than allowing a nullable placeholder.

Do not add `std::shared_ptr<runtime_state> runtime = nullptr` merely to avoid updating constructors. A nullable runtime would infect every operation with a state that the accepted design says cannot exist.

## 2.8 Turning `tensor` Into A Handle

After the identity-only implementation exists, the public class owns one pointer:

```cpp
#include "comtam/tensor/impl.h"

namespace comtam {

class tensor {
   public:
    tensor(const tensor&) = default;
    tensor& operator=(const tensor&) = default;
    ~tensor() = default;

   private:
    explicit tensor(std::shared_ptr<tensor_impl> impl)
        : impl_(std::move(impl)) {
        COMTAM_CHECK_AND_THROW(impl_, std::runtime_error,
                               "tensor requires an implementation");
    }

    std::shared_ptr<tensor_impl> impl_;
};

}  // namespace comtam
```

No public default constructor is added. Every usable tensor has one non-null implementation.

The explicit copy operations document the semantic promise: copying a handle shares identity. They perform the default `shared_ptr` copy; they do not clone `tensor_impl`.

Do not add custom move operations in the first version. A default move would leave the source handle with a null `impl_`, weakening the non-null public invariant and forcing every method to define moved-from behavior. C++17 copy elision handles ordinary return-by-value, and copying one `shared_ptr` is acceptable for this educational runtime. Move semantics can be introduced later with an explicit contract if measurement or container usage earns them.

## 2.9 Centralizing Implementation Construction

The current class repeats dtype/view/storage initialization across several constructors. Once those values live in `tensor_impl`, centralize the final assembly step:

```cpp
static std::shared_ptr<tensor_impl> make_impl(
    DType dtype,
    view logical_view,
    std::shared_ptr<core::storage> storage) {
    return std::make_shared<tensor_impl>(
        dtype, std::move(logical_view), std::move(storage));
}
```

Public constructors remain responsible for semantic validation, allocation, and upload. `make_impl` is responsible only for assembling an already-valid logical tensor. Do not move shape validation or dtype dispatch into `tensor_impl`; it is an ownership object, not a second tensor API.

For example, an existing allocation constructor changes conceptually from:

```cpp
tensor::tensor(const view_vector& shape, metal_device& device, DType dtype)
    : dtype_(dtype), view_(shape), storage_(allocate(...)) {}
```

to:

```cpp
tensor::tensor(const view_vector& shape, metal_device& device, DType dtype) {
    view logical_view(shape);
    auto buffer = device.allocate(bytes_for(dtype, logical_view));
    auto storage = std::make_shared<core::storage>(std::move(buffer));
    impl_ = make_impl(dtype, std::move(logical_view), std::move(storage));
}
```

`bytes_for` is pseudocode for the existing dtype-dispatched byte calculation, not a requirement to invent a new abstraction. Keep the current direct calculation unless repetition has already earned a helper.

## 2.10 Rewriting Access Without Duplicating State

After `impl_` exists, remove the legacy fields completely:

```cpp
// Remove these. Do not keep them as a transition cache.
DType dtype_;
view view_;
std::shared_ptr<core::storage> storage_;
```

Keeping both representations creates two sources of truth. A constructor or movement operation will eventually update one and forget the other.

Getters become projections from the implementation:

```cpp
size_int numel() const {
    return static_cast<size_int>(impl_->logical_view.numel());
}

view_vector shape() const {
    return impl_->logical_view.shape;
}

DType dtype() const {
    return impl_->dtype;
}
```

Host reads, host writes, checks, and dispatch preparation must also read through `impl_`. The compiler is useful here: remove the old fields first, then let every unresolved `dtype_`, `view_`, and `storage_` identify a migration site.

Do not create references such as `DType& dtype_ = impl_->dtype` to avoid editing callers. Reference data members complicate assignment and merely hide the duplicate representation problem.

## 2.11 Copy Semantics Become Structural

After migration:

```cpp
tensor b = a;
```

produces this graph:

```text
a handle ─┐
          ├── shared tensor_impl
b handle ─┘      ├── dtype
                  ├── logical view
                  └── shared storage -> MTL::Buffer
```

Destroying either handle decrements the implementation reference count. The implementation and storage remain alive while the other handle survives.

Copy assignment has a stronger consequence:

```cpp
b = a;
```

`b` releases its previous logical identity and becomes another handle to `a`. This is correct C++ handle assignment, but a future optimizer must not update a registered parameter by assigning a new tensor handle; it must mutate the existing parameter's storage while preserving its implementation identity. That constraint will matter in the optimizer track.

## 2.12 Movement Creates A New Implementation

Public movement operations share bytes but create logical tensor identity:

```cpp
tensor tensor::reshape(const view_vector& shape) const {
    auto next_view = impl_->logical_view.reshape(shape);
    auto next_impl = make_impl(impl_->dtype,
                               std::move(next_view),
                               impl_->storage);
    return tensor(std::move(next_impl));
}
```

The resulting ownership graph is:

```text
a handle -> impl A ─┐
                    ├── shared storage -> MTL::Buffer
c handle -> impl C ─┘

impl A.view = original view
impl C.view = reshaped view
```

Even a metadata no-op such as reshaping `(2, 3)` to `(2, 3)` should follow the accepted public movement rule and create a new implementation. Returning `*this` would silently remove a semantic movement edge once recording exists.

Apply the same pattern to `permute`, `transpose`, `shrink`, and `expand`. Each validates or derives its new `view`, then creates one new implementation sharing storage.

## 2.13 Raw Internal Aliases Are Not Tensor Identities

Binary broadcasting currently constructs temporary tensors from expanded views:

```cpp
tensor a_expand = tensor(a.storage_, a.view_.expand(final_shape), a.dtype_);
```

Once every constructed tensor receives an identity, that line would create a logical tensor for an alias that exists only to describe kernel indexing. Later recording could mistake it for a public `expand` operation.

Keep internal aliases as raw metadata:

```cpp
view a_expanded_view = a.impl_->logical_view.expand(final_shape);
view b_expanded_view = b.impl_->logical_view.expand(final_shape);

core::command_desc cmd = {
    .a = {
        .storage = a.impl_->storage.get(),
        .view = core::view_desc::from_view(a_expanded_view),
    },
    .b = {
        .storage = b.impl_->storage.get(),
        .view = core::view_desc::from_view(b_expanded_view),
    },
    // output omitted here
};
```

The raw views live only for dispatch preparation. They do not own storage, runtime, or autograd state. The original operand implementations remain the semantic inputs.

This distinction is not an optimization detail. It determines the future graph:

```text
correct:   add output -> add_backward -> original a, original b
incorrect: add output -> hidden expand nodes -> temporary alias identities
```

## 2.14 Proving Identity Without Using Value Equality

Chapter 2 needs direct identity tests before gradients exist. C++ stack addresses and storage pointers are insufficient, so add one narrow identity query during the learning implementation:

```cpp
bool is_same(const tensor& other) const noexcept {
    return impl_ == other.impl_;
}
```

The name must communicate identity, not value comparison. Do not overload `operator==` for this purpose because users will reasonably expect tensor value equality or an elementwise comparison later.

Whether `is_same` remains public is a Module 10 API decision. It is useful now because it makes the semantic contract executable and will also help parameter-identity tests.

Storage sharing needs a different query. Avoid exposing raw mutable storage merely for tests. A narrow internal or test-support method can compare `impl_->storage` ownership without returning a writable pointer.

## 2.15 Identity Tests Before Runtime Binding

The identity-only checkpoint should prove:

```cpp
TEST_CASE("copying a tensor shares logical identity") {
    tensor a = make_tensor(...);
    tensor b = a;

    REQUIRE(a.is_same(b));
}

TEST_CASE("independent construction has distinct identity") {
    tensor a = make_tensor({1.0f, 2.0f});
    tensor b = make_tensor({1.0f, 2.0f});

    REQUIRE_FALSE(a.is_same(b));
}

TEST_CASE("public movement creates identity and shares storage") {
    tensor a = make_tensor(...);
    tensor b = a.reshape(...);

    REQUIRE_FALSE(a.is_same(b));
    REQUIRE(test_access::shares_storage(a, b));
}
```

Test every public movement operation, not only reshape. The view metadata assertions already present in movement tests remain valuable; identity assertions add a different claim.

Also test handle lifetime:

```cpp
tensor survivor = [&] {
    tensor original = make_tensor(...);
    tensor copy = original;
    return copy;
}();

REQUIRE(expected_values(survivor));
```

This does not prove implementation identity by itself, but it proves the shared owner survives the original handle.

## 2.16 Adding Runtime Identity To `tensor_impl`

After Chapter 1 and the identity-only checkpoint pass, add runtime identity as a required member:

```cpp
struct tensor_impl {
    tensor_impl(std::shared_ptr<core::runtime_state> runtime_value,
                DType dtype_value,
                view view_value,
                std::shared_ptr<core::storage> storage_value)
        : runtime(std::move(runtime_value)),
          dtype(dtype_value),
          logical_view(std::move(view_value)),
          storage(std::move(storage_value)) {
        COMTAM_CHECK_AND_THROW(runtime, std::runtime_error,
                               "tensor_impl requires a runtime");
        COMTAM_CHECK_AND_THROW(storage, std::runtime_error,
                               "tensor_impl requires storage");
    }

    std::shared_ptr<core::runtime_state> runtime;
    DType dtype;
    view logical_view;
    std::shared_ptr<core::storage> storage;
};
```

Declare `runtime` before `storage`. Members are destroyed in reverse declaration order, so storage releases its Metal buffer before the final runtime reference can destroy the device and queue.

The runtime and storage must correspond: storage must have been allocated through that runtime's device. The first implementation enforces this by making tensor factories the only construction path and never accepting arbitrary public runtime/storage pairs. `core::storage` does not currently record runtime identity, so the invariant is established structurally at allocation rather than rechecked inside `tensor_impl`.

Do not expose a public constructor that accepts both an arbitrary `runtime_state` and arbitrary storage. That would make an invalid pair representable without adding any learning value.

## 2.17 Constructor Migration Is A Real API Change

Current tensor constructors accept `core::metal_device&`. A device reference alone cannot provide the shared runtime state that the tensor must retain. There is no safe way to recover the owning `runtime_state` from an arbitrary device reference.

This means the assumption "we can add tensor-bound runtime without changing constructors" is false. The concrete consequence would be tensors with missing or guessed runtime identity. The better alternative is to migrate construction to `core::context&` while preserving the established dtype position and default construction path.

The explicit form becomes conceptually:

```cpp
tensor::tensor(const float* data,
               const view_vector& shape,
               DType dtype,
               const core::context& context) {
    view logical_view(shape);
    auto buffer = context.device().allocate(bytes_for(dtype, logical_view));
    context.device().copy(data, logical_view.numel(), buffer);
    auto storage = std::make_shared<core::storage>(std::move(buffer));

    impl_ = std::make_shared<tensor_impl>(
        context.state(), dtype, std::move(logical_view), std::move(storage));
}
```

The accepted public ordering is:

```cpp
template <typename T>
explicit tensor(const T* data,
                const view_vector& shape,
                DType dtype = DType::Float32,
                const core::context& context = core::default_context());
```

This supports a progressive call surface:

```cpp
tensor(data, shape);                          // default dtype and runtime
tensor(data, shape, DType::Float32);          // explicit dtype, default runtime
tensor(data, shape, DType::Float32, context); // explicit dtype and runtime
```

Putting `context` last preserves the established `tensor(shape, dtype)` form and keeps the less commonly changed runtime argument trailing. The tradeoff is deliberate: selecting a non-default context also requires spelling the dtype. That small verbosity is preferable here to inserting runtime selection between shape/value metadata and dtype or maintaining a larger overload family. Apply the same order consistently to scalar, shape, and view construction.

Adapt the existing dtype-dispatching template rather than replacing it with float-only duplicated code. The sketch isolates runtime ownership; it does not supersede current dtype validation.

Every constructor path must choose a runtime, including scalar construction, empty output allocation, host-upload construction, and internal operation results. There must be no legacy `metal_device&` path that creates a tensor without runtime identity.

## 2.18 Inferring Runtime In Operations

After tensors retain runtime state, a unary operation needs no context parameter:

```cpp
static tensor neg(const tensor& input) {
    return uop(input, Op::NEG, input.impl_->runtime);
}
```

A binary operation first validates runtime identity:

```cpp
static std::shared_ptr<core::runtime_state> common_runtime(
    const tensor& lhs, const tensor& rhs) {
    COMTAM_CHECK_AND_THROW(
        lhs.impl_->runtime == rhs.impl_->runtime,
        std::runtime_error,
        "operands belong to different runtimes");
    return lhs.impl_->runtime;
}
```

The internal dispatch helper should receive the shared runtime handle, not look up `default_context()`:

```cpp
static tensor bop(const tensor& lhs,
                  const tensor& rhs,
                  Op op,
                  const std::shared_ptr<core::runtime_state>& runtime);
```

This handle both supplies `device()` and `kernels()` and becomes part of the output implementation.

Composed operations must create constants in the operand runtime. For example, `mean` cannot use the default runtime blindly when its input belongs to an explicit runtime:

```text
scale runtime = input runtime
sum result runtime = input runtime
mul operands = same runtime
```

## 2.19 A Safe Compatibility Transition

You may temporarily retain explicit-context operation overloads while migrating call sites, but they must validate rather than override tensor identity:

```cpp
static tensor add(const tensor& lhs,
                  const tensor& rhs,
                  core::context& context) {
    auto runtime = common_runtime(lhs, rhs);
    COMTAM_CHECK_AND_THROW(
        runtime == context.state(),
        std::runtime_error,
        "explicit context does not own the operand runtime");
    return bop(lhs, rhs, Op::ADD, runtime);
}
```

This overload is a compatibility facade. Passing another context must not silently dispatch the tensors through that context.

Once context-free calls and tests are established, remove or deliberately retain explicit overloads according to the public API goal. Do not keep two independent implementations.

## 2.20 Runtime And Storage Ownership Traces

### Ordinary Copy

```text
context handle ─┐
                ├── runtime_state -> device/kernel library
a tensor ───────┤
b tensor ───────┘
  both tensors -> same tensor_impl -> same storage
```

Destroying the original context handle does not destroy runtime state while either tensor survives.

### Public Movement

```text
a -> tensor_impl A ─┬── runtime_state
                    └── storage

b -> tensor_impl B ─┬── same runtime_state
                    └── same storage
```

The implementations have different views and later will have different autograd metadata.

### Eager Allocating Operation

```text
a -> impl A -> runtime R, storage A
b -> impl B -> runtime R, storage B

add(a, b)
  -> impl C -> runtime R, new storage C
```

The output does not need its input storage for forward lifetime after synchronous dispatch, but future backward nodes will retain the required input implementations.

### Destruction Order

When the final tensor handle releases an implementation:

```text
future autograd metadata and producer release first
storage releases its MTL::Buffer
runtime shared_ptr releases last
runtime destroys kernel library before device when it is the final owner
```

Member declaration order must preserve this dependency rather than relying on accidental external context lifetime.

## 2.21 Where Future Autograd Metadata Belongs

The final implementation will extend the same object:

```cpp
struct tensor_impl {
    std::shared_ptr<core::runtime_state> runtime;
    DType dtype;
    view logical_view;
    std::shared_ptr<core::storage> storage;
    autograd_meta autograd;  // Chapter 3-4
};
```

Do not define `autograd_meta` in Chapter 2. Its representation depends on recording states, producer ownership, leaf gradients, and consumed graphs that later chapters must derive.

The important decision is already made: autograd metadata belongs to logical tensor identity, so it lives inside `tensor_impl`, not beside it in each public handle.

This also avoids a recursive-value mistake. `tensor_impl` cannot contain a `tensor` value directly because `tensor` owns `tensor_impl`. Later chapters will use forward declarations and pointer handles for producer nodes and gradients where recursion is required.

## 2.22 Incremental Implementation Plan

### Checkpoint A: Clean The Draft

1. Add the direct `<memory>` include.
2. Rename the member from `view` to `logical_view`.
3. Add a constructor that rejects null storage.
4. Do not add runtime or autograd fields yet.

### Checkpoint B: Convert `tensor` Into A Handle

1. Replace the three legacy fields with `std::shared_ptr<tensor_impl> impl_`.
2. Centralize implementation construction.
3. Rewrite every constructor and getter through `impl_`.
4. Rewrite host access and dispatch preparation through `impl_`.
5. Keep existing explicit-context operation signatures temporarily.
6. Build and run the entire forward suite.

### Checkpoint C: Establish Movement Identity

1. Make every public movement allocate a new implementation sharing storage.
2. Replace binary broadcast temporary tensors with raw local views.
3. Add identity and storage-sharing tests for every movement.
4. Re-run broadcast and non-contiguous forward tests.

### Checkpoint D: Bind Runtime

1. Complete Chapter 1's shared `runtime_state` refactor.
2. Add required runtime state to `tensor_impl`.
3. Migrate tensor construction from `metal_device&` to explicit/default context.
4. Make operation outputs inherit operand runtime.
5. Add cross-runtime rejection before allocation or dispatch.
6. Add context-free APIs and compatibility validation as needed.
7. Run the complete forward suite again.

Do not begin recording or gradient rules until Checkpoint D passes.

## 2.23 Test Matrix

The chapter gate needs distinct evidence for identity, aliasing, runtime, and unchanged forward behavior:

| Area | Required evidence |
| --- | --- |
| Copy | Copy construction and copy assignment share logical identity |
| Independent values | Equal independently constructed tensors have distinct identity and storage |
| Movement | Each public movement has distinct identity, shared storage, correct derived view, same runtime |
| Allocating ops | Unary, binary, reduction, and matmul results have new identity/storage and inherited runtime |
| Runtime lifetime | Tensor remains usable after its user-visible context handle is destroyed |
| Runtime mismatch | Mixed-runtime binary operations reject before dispatch |
| Constructor selection | Explicit construction retains the selected trailing context; omitted context uses the stable default |
| Internal aliases | Binary broadcasting uses raw view metadata and preserves existing forward results |
| Regression | Complete existing CTest suite passes after each checkpoint |

Use direct identity assertions for identity and MLX-C comparisons for forward values. One kind of test cannot substitute for the other.

## 2.24 Common Mistakes

### Using Storage As Identity

This merges views and destroys the ability to record movement edges. Storage owns bytes; `tensor_impl` owns logical interpretation and autograd identity.

### Deep-Copying `tensor_impl` In The Tensor Copy Constructor

```cpp
tensor(const tensor& other)
    : impl_(std::make_shared<tensor_impl>(*other.impl_)) {}  // wrong
```

This creates a second identity. Default-copy the `shared_ptr` instead.

### Keeping Legacy Fields Beside `impl_`

Two representations inevitably diverge. Remove the old fields and let compiler errors enumerate every migration site.

### Giving `tensor_impl` A Default Constructor

A null runtime or storage state is not a supported tensor. Make invalid construction impossible instead of checking it throughout the runtime.

### Recovering Runtime From `metal_device&`

A device does not identify which shared runtime state owns it. Change the construction boundary to context/runtime handles.

### Looking Up The Default Runtime During Operations

Only construction chooses a default. Existing tensors already carry their runtime; using the current default would make behavior depend on unrelated global selection.

### Making Internal Broadcast Aliases Into Tensors

This creates semantic identities for physical submission metadata and later produces duplicate graph edges.

### Returning `*this` For A No-Op Movement

This may look efficient, but it changes the accepted movement identity rule. Optimization cannot silently erase semantic nodes.

### Adding Autograd Fields During The Identity Refactor

If a test fails, you will not know whether copy semantics, view aliasing, graph ownership, or gradient logic caused it. Finish identity first.

## 2.25 Your Implementation Exercise

Implement this chapter in the four checkpoints above. After each checkpoint, write a short answer to these questions:

1. Which object defines logical tensor identity at this checkpoint?
2. Which objects are shared by `tensor b = a`?
3. Which objects are shared by `auto b = a.reshape(...)`?
4. What keeps the runtime alive after an explicit context handle is destroyed?
5. Which construction path would still permit a tensor without runtime identity?
6. Does binary broadcasting construct any hidden tensor identity?
7. Which exact tests distinguish logical identity from storage sharing?

When a compiler error appears during migration, classify it before fixing it: constructor assembly, metadata access, storage access, runtime inference, or public API call site. This turns the large mechanical refactor into an ownership checklist.

## 2.26 Chapter Completion Gate

Chapter 2 is complete only when all of these statements are true:

- Every public tensor handle owns exactly one non-null shared `tensor_impl`.
- Copy construction and copy assignment share one logical identity.
- Every public movement creates a new identity while sharing storage and runtime.
- Every allocating operation creates a new identity and storage in the operand runtime.
- `tensor_impl` owns runtime, dtype, logical view, and storage interpretation together.
- No legacy dtype/view/storage fields remain on the public tensor handle.
- No tensor construction path lacks runtime identity.
- Mixed-runtime operands reject before allocation or dispatch.
- Internal broadcast aliases remain raw metadata and never create tensor identity.
- Context destruction cannot invalidate surviving tensors.
- Identity, aliasing, runtime, and forward-correctness tests all pass.
- No recording state, producer node, or gradient slot has been added yet.

## Agent Feedback / Grading

Status: passed in isolation; Chapter 3 remains blocked by Chapter 1's two missing direct context assertions.

The refactor now gives `tensor` one non-null shared implementation identity, preserves that identity under copy construction and assignment, and creates distinct implementations for public movement while sharing storage and runtime. Allocating unary, binary, reduction, and matmul operations create fresh storage in the operand runtime. Mixed-runtime binary operations reject before dispatch, composed `mean` keeps its scale in the operand runtime, and binary broadcasting uses raw temporary views rather than hidden tensor identities.

The rewritten tests are trustworthy for the claims they actually make. `tests/tensor/identity.cpp` directly distinguishes handle, implementation, storage, runtime, and value identity; covers every public movement; checks allocating operations; exercises context-handle destruction; and rejects mixed runtimes across `add`, `sub`, `mul`, `div`, and `matmul`. Existing MLX-backed forward tests remain the independent numerical oracle. The phrase "reject before dispatch" is established jointly by those rejection tests and source inspection of validation order; the tests alone do not instrument submission counts.

Independent verification on the current revision:

```text
cmake --build build --target comtam_tests -j 6  -> compiled
ctest --test-dir build --output-on-failure      -> passed, 51/51 outside the sandbox
git diff --check                                -> passed
```

Do not begin Chapter 3 until you can draw ordinary copy, movement, allocating operation, and destruction graphs from memory and explain why storage identity cannot replace logical tensor identity.

## Reflection Before Chapter 3

After implementing the chapter, record where the migration was harder than the design predicted. In particular, note whether constructor migration, raw broadcast aliases, or runtime inference exposed an invariant missing from this guide. That evidence will determine how Chapter 3 introduces recording without hiding another cross-cutting refactor.
