# Chapter 3: Recording And `no_grad`

**Status:** Ready to implement. Chapters 1 and 2 are the prerequisites.

Chapter 2 established what a logical tensor is. This chapter adds the first piece of autograd state to it, and answers the question that has to be settled before any node type exists: when should an eager operation record anything at all?

## The Concrete Problem

Every operation in [tensor.cpp](../../../comtam/tensor/tensor.cpp) does exactly three things today: validate, allocate an output, dispatch a kernel. Autograd needs a fourth step for *some* operations - attach a producer node to the result.

"Some" is the entire difficulty. Three callers must not record, and they are not exotic:

- **Inference.** Running a trained model to get predictions should not build a graph. If it does, memory grows with every batch and nothing ever frees it, because the graph retains its inputs.
- **The optimizer.** `w = w - lr * grad` is arithmetic on tensors that require gradients. Recording it would make the parameter update part of the next backward pass, which is nonsense - and it would make `w` a non-leaf, destroying the gradient slot the optimizer just read.
- **Backward itself.** Every backward rule computes with the same public primitives. If those calls record, the first backward pass builds a second graph, which builds a third. comtam defers higher-order gradients, so backward must run with recording off.

So the operation needs an answer to two separate questions before it can decide:

> Does any operand actually require a gradient, and is recording currently enabled here?

Both must be true. Neither alone is sufficient.

## The Mental Model

Recording is the conjunction of a *data* property and a *scope* property.

```text
record = (any operand requires grad) AND (recording enabled in this scope)
```

The data property travels with tensors. The scope property travels with the call stack. They need completely different mechanisms, and conflating them is the classic mistake - a single global `grad_enabled` bool tries to be both and is correctly neither.

## Part 1: The Data Property

`requires_grad` describes a logical tensor, not a handle and not a buffer. Chapter 2 already decided where that goes: `tensor_impl` ([impl.h:25](../../../comtam/tensor/impl.h:25)). One `bool`, default `false`.

The consequences follow from Chapter 2's ownership graph without any new rules:

```cpp
tensor b = a;              // same impl -> b.requires_grad() == a.requires_grad(), always
auto  c = a.reshape({4});  // new impl -> c's flag is whatever the operation sets
```

That second line is the trap. Movement operations build a *new* `tensor_impl` ([tensor.h:157](../../../comtam/tensor/tensor.h:157)), so the flag does not come along by itself. Every one of `permute`, `transpose`, `shrink`, `expand`, and `reshape` has to propagate it explicitly. Miss one and a gradient silently stops flowing through that operation - the forward result is numerically perfect, and backward just quietly produces nothing. There is no test that catches this except one that checks the flag directly.

Two propagation rules cover everything:

- A **leaf** requires a gradient because the user said so. Setting the flag is not an operation and is not affected by scope.
- A **result** requires a gradient when recording is enabled and at least one input requires one.

Note the asymmetry: `requires_grad(true)` on a leaf works fine inside a suppressed scope. Turning a parameter into something trainable is bookkeeping, not computation. Only *results* are affected by scope. This matches PyTorch, and it matters because model construction often happens inside inference-shaped code.

One exception to enforce now, before it can silently produce wrong numbers: `max` has no tie policy and no gradient test, so [`ARCHITECTURE.md`](../../ARCHITECTURE.md) requires recording to reject it. Propagating `requires_grad` through `max` must throw rather than produce a tensor that claims a gradient nobody can compute. Composed `min` inherits the restriction through `neg(max(neg(a)))`.

## Part 2: The Scope Property

Now the harder half. What object holds "recording is off right now", and how far does "right now" reach?

**A global `bool`.** Not nestable: an inner suppressed scope re-enables recording when it exits, even though an outer scope wanted it off. Not thread-safe. Not scoped to a runtime.

**A global counter.** Nesting works - increment on entry, decrement on exit, enabled when zero. Still a data race across threads, and still process-wide.

**A `thread_local` counter.** Nesting and thread isolation both work, and this is genuinely most of the value. What it still gets wrong: a `no_grad` scope for one runtime silences operations on every other runtime in that thread. comtam's tests construct isolated contexts specifically to prove runtimes are independent ([tests/core/context.cpp:31](../../../tests/core/context.cpp:31)); process-wide recording state punches a hole straight through that isolation. A future harness that runs an oracle runtime beside a training runtime would hit the same hole.

**A `thread_local` depth per runtime.** The accepted design. Thread-local for isolation, a counter for nesting, keyed by runtime for the reason above.

Keep the container boring. A process has one or two runtimes, so a small vector scanned linearly beats a hash map on both code size and speed:

```cpp
// in an anonymous namespace, one per thread
thread_local std::vector<std::pair<core::runtime_state*, int>> suppression_depth;
```

Two properties this has to guarantee, and both come from the guard rather than the container:

- **The key cannot dangle.** The guard holds a `shared_ptr<runtime_state>` for its whole lifetime, so the runtime cannot be destroyed while its raw pointer is a live key.
- **The table cannot grow forever.** When a depth returns to zero, erase the entry. Otherwise every runtime ever suppressed leaves a permanent row, and a destroyed runtime's address can be recycled by a new allocation - a stale zero entry would then be harmless, but a stale *non-zero* one would silently suppress a fresh runtime.

## The Guard

RAII, not a pair of free functions:

```cpp
class no_grad {
   public:
    explicit no_grad(const core::context& context);
    ~no_grad();

    no_grad(const no_grad&) = delete;
    no_grad& operator=(const no_grad&) = delete;

   private:
    std::shared_ptr<core::runtime_state> runtime_;
};
```

`enable_grad()`/`disable_grad()` free functions would leak suppression whenever a scope exits early - and backward rules throw. The whole point is that a throw between the two calls must not leave recording permanently off. Non-copyable for the same reason it is RAII: two guards sharing one increment would decrement twice.

The constructor argument is deliberately not defaulted, and this is worth a moment. `no_grad guard;` meaning "the default context" reads beautifully and fails silently in exactly the case runtime-specificity exists to handle: a user whose tensors live on an explicit context writes `no_grad guard;`, gets the *default* runtime's counter incremented, and their operations record anyway. A verbose guard that always works beats a terse one that quietly does nothing. If call sites prove awkward in practice, add an overload taking a tensor - `no_grad guard(some_param);` - rather than a default argument.

## Where The Check Goes

One query function, called by each dispatch path:

```cpp
bool should_record(const std::shared_ptr<core::runtime_state>& runtime,
                   std::initializer_list<const tensor_impl*> inputs);
```

The four allocating operations ([tensor.cpp:20](../../../comtam/tensor/tensor.cpp:20), [59](../../../comtam/tensor/tensor.cpp:59), [86](../../../comtam/tensor/tensor.cpp:86), [119](../../../comtam/tensor/tensor.cpp:119)) already establish `runtime_state` and validate their operands. The recording decision belongs after validation - an operation that is about to reject mixed runtimes has no business consulting a graph flag first - and its only effect in this chapter is setting one bool on the output implementation.

Compositions need no special handling, and that is the payoff of the primitive-surface work. `sub` is `add(a, neg(b))` ([tensor.h:196](../../../comtam/tensor/tensor.h:196)); each primitive inside decides for itself and the flag propagates through. Do not add a recording decision to `sub`, `div`, `mean`, or `min`.

## `detach()`

Chapter 2's semantic matrix left one row unimplemented, and this is its chapter. `detach()` returns a tensor with the same storage, view, dtype, and runtime, and a *new* implementation whose `requires_grad` is false.

It exists because "read these values without extending the graph" is a real need - logging a loss, feeding a value to an oracle comparison, snapshotting a parameter. It is also the cleanest proof that logical identity is not storage identity: everything observable about the bytes is shared, and the two tensors are still not the same logical tensor.

`detach` shares storage, so writing through one detached handle is visible through the other. That is a Chapter 6 problem - it is exactly what storage versions exist to catch - and it should be documented now rather than discovered later.

## Implementation Checkpoints

Do these in order. Each is independently testable, which is the point.

**A. The flag.** Add `requires_grad` to `tensor_impl` with a public getter and setter on `tensor`. Nothing propagates yet. Prove copy shares it and independent construction does not.

**B. Propagation through movement.** Make all five movement operations carry the flag to the new implementation. This is the checkpoint most likely to be silently incomplete, so test all five, not a representative one.

**C. The guard.** Implement the thread-local per-runtime depth and `no_grad`. Still nothing consults it. Test nesting, exception safety, thread isolation, and runtime isolation on the counter alone - the mechanism is easier to debug before any operation depends on it.

**D. Propagation through operations.** Wire `should_record` into the four dispatch paths, propagate to the output implementation, and reject recording through `max`.

**E. `detach()`.**

Resist merging C and D. If recording suppression and flag propagation land together, a failing test cannot tell you which half is wrong.

## Tests That Prove The Mechanism

The flag itself:

- A copy observes the leaf's flag; independently constructed tensors do not.
- Each of the five movements propagates it, checked one at a time.
- `add`, `mul`, `neg`, `recip`, `sum`, and `matmul` propagate it from either operand.
- Compositions `sub`, `div`, `mean`, and `min` propagate it without owning a decision.
- `max` with a gradient-requiring operand throws.

The guard:

- Nested guards suppress until the *outer* one exits.
- A guard destroyed by an exception restores the previous depth.
- Recording stays enabled on runtime B while a guard is active on runtime A.
- A guard on one `std::thread` does not affect an operation on another. This is the test that fails if the counter is `static` rather than `thread_local`, and nothing else catches that.
- Setting `requires_grad(true)` on a leaf still works inside a guard.
- Results created inside a guard do not require gradients even when their inputs do.

`detach`:

- Shares storage and runtime, has distinct logical identity, does not require a gradient.
- Values match the source exactly.

Numerical correctness is not at stake in this chapter - every forward result is bit-identical to before. Every test here is about flags, identity, and scope, so write them that way rather than comparing values against the oracle.

## Common Mistakes

**A `bool` where a counter belongs.** Nesting silently breaks: the inner scope's exit re-enables recording inside the outer scope. The forward numbers stay right and the graph is quietly wrong.

**`static` instead of `thread_local`.** Works perfectly in every single-threaded test.

**Free `disable_grad()`/`enable_grad()` functions.** One throw between them and recording is off for the rest of the process.

**A guard holding a raw `runtime_state*`.** The key outlives the runtime, and address reuse turns that into suppression of an unrelated runtime.

**Forgetting one movement operation.** No forward test can see it. Only a direct flag assertion can.

**Consulting the flag before validation.** Mixed-runtime and shape errors must still reject first; recording state is not an excuse to reorder checks.

**Putting the depth counter in `runtime_state`.** It is reachable and it is tempting, and it is the hidden global state Chapter 1 refused. A non-static member cannot be `thread_local` anyway, so it would also be wrong.

**Using `no_grad` to suppress errors.** It suppresses recording. Validation, dtype rules, and runtime isolation are unaffected.

## Your Implementation Exercise

Work checkpoints A through E in order, writing each checkpoint's tests before starting the next.

Two decisions are yours to make and record:

1. Whether `no_grad` takes a `context`, a `tensor`, or both. The chapter recommends explicit `context` and explains why a defaulted argument is a trap; if you disagree, write down the call sites that made you disagree.
2. Whether `requires_grad` is settable on a non-leaf. The simplest answer - throw, because a non-leaf's flag is derived - is probably right, but state it explicitly and test it either way.

Before writing code, draw the decision for one concrete case: `tensor::mul(a, b)` where `a` requires a gradient, `b` does not, and an outer `no_grad` guard is active on a *different* runtime. Name every object consulted and the order in which they are consulted.

## Chapter Completion Gate

- `requires_grad` lives on `tensor_impl` and is observable through the public handle.
- Copies share the flag structurally; every movement propagates it explicitly.
- Every allocating operation propagates it from its operands.
- Recording suppression is lexical, nestable, thread-local, and runtime-specific.
- Suppression survives exceptions and cannot leak past its scope.
- The suppression table cannot hold a dangling key or grow without bound.
- Setting a leaf's flag works regardless of scope.
- Recording through `max` is rejected.
- `detach()` returns a distinct logical identity over shared storage with no gradient requirement.
- No node type, gradient slot, or backward function exists yet.

## What Comes Next

Chapter 4 uses the decision this chapter produces. Once an operation knows it should record, it needs somewhere to put the rule - a producer node owned by the output, holding its parents and its saved values without retaining its own output. That ownership constraint is the whole chapter, because getting it wrong produces a reference cycle that never frees a graph.
