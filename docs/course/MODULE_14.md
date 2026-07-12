## Module 14: Python API

Expose the stable C++ runtime to Python without creating a second ownership,
autograd, or operator implementation.

## Why Python Comes This Late

Bindings multiply the cost of every unstable semantic decision. Tensor identity,
context ownership, exceptions, async synchronization, state dictionaries, and
supported operations must already be explicit so Python can wrap them rather
than invent replacements.

Magnetron is the architectural reference: a native core with a thin modern
Python layer. Borrow that separation, but preserve comtam's explicit-context rule
and Metal-only scope.

## Module Contract

**Prerequisites:** Modules 10-13 pass; the public C++ API is versioned, persistence
works, and performance/lifetime baselines exist.

**You will produce:** a compiled Python extension, a small pure-Python package,
bindings for context/tensor/autograd/nn/SGD/state dictionaries, NumPy copy
interop, exception translation, type hints, and Python conformance tests.

**Supported scope:** CPython on supported Apple Silicon/macOS versions, float32,
one explicit runtime context, and the existing C++ feature set.

**Not required:** Python-defined custom autograd functions, arbitrary NumPy
strided zero-copy ownership, DLPack, CUDA, Torch compatibility, or API parity
with Magnetron/PyTorch.

**Recommended binding tool:** nanobind, because Magnetron provides a nearby
reference and it integrates cleanly with CMake. pybind11 is acceptable if the
solution note gives a concrete project reason; do not write a custom CPython C
extension for this course.

## Python Ownership Model

Do not introduce an invisible process-wide Metal singleton for convenience.
The minimum safe model is:

```text
comtam.Context
  owns shared native context state

comtam.Tensor
  owns a native tensor handle
  holds shared native Context state from the Context that created it

operations
  require operands from the same Context
  call the native C++ implementation exactly once
```

A context-manager syntax may make lifetime explicit, but leaving a `with` block
must not invalidate tensors that still retain the context unless that behavior
is deliberately documented and tested.

The decisive lifetime test drops the last Python `Context` object while a tensor,
a view, a gradient, and queued native work still exist. Those wrappers must keep
the shared native context state alive through their documented completion and
readback boundaries.

## Assignment 14.1: Scaffold The Extension And Package ⭐⭐

**Task:** create a private compiled module such as `comtam._comtam` and a public
pure-Python package `comtam`. The public package re-exports supported names and
contains higher-level composition only where it adds no tensor semantics.

The extension must link the same native library exercised by C++ tests. Do not
copy kernel dispatch, shape logic, or gradient formulas into bindings.

**Smoke test:** in a clean Python environment, import `comtam`, create a context
and tensor, run an add, and read the result.

## Assignment 14.2: Bind Context, Tensor, And Views ⭐⭐⭐

**Task:** expose construction, shape/dtype metadata, supported movement ops,
forward ops, `requires_grad`, `.grad`, `detach`, `backward`, and a scoped
`with ctx.no_grad():` recording guard.

Python object identity must follow the native identity rules:

- assigning `b = a` refers to the same Python/native handle
- an op result is a new tensor identity
- a view result shares storage according to Module 10
- `.grad` returns the native leaf gradient, not a copied Python-only cache
- tensors from different contexts fail before dispatch
- `no_grad` restores the previous recording mode after normal or exceptional
  scope exit and affects only the owning context/execution scope

**Lifetime test:** delete the user-visible `Context`, retain only tensor/view/grad
wrappers, complete queued work, and read correct values. Then destroy the final
wrappers and prove the native context state is released.

Do not expose mutable raw storage pointers. Represent shapes as immutable Python
tuples and dtypes as stable named objects or enums.

## Assignment 14.3: Add NumPy Copy Interop ⭐⭐⭐

**Task:** support explicit copying in both directions:

```python
ctx.tensor(numpy_array)     # copy into comtam-owned storage
tensor.numpy()             # synchronize and return an owning NumPy array
```

Start with C-contiguous float32 arrays. For non-contiguous arrays, either make an
explicit contiguous copy or reject them with a clear error. State which behavior
you chose. Returned arrays must remain valid after the tensor/context is
destroyed, which is why copy semantics come first.

Zero-copy shared-buffer interop is deferred until ownership, mutation, and async
lifetime can be specified independently.

## Assignment 14.4: Translate Errors And Manage The GIL ⭐⭐⭐

**Task:** map native validation/runtime exceptions to appropriate Python
exceptions while retaining the native operation context. A shape error should
not become an uninformative generic failure.

Release the GIL around blocking native work such as synchronization/readback and
long training operations, but retain Python objects/native handles before
release. Never call Python APIs without the GIL.

**Tests:** invalid shapes/dtypes/axes, cross-context operands, Metal error
propagation, concurrent independent Python threads where supported, and object
destruction while native work is in flight.

## Assignment 14.5: Bind Autograd, `nn`, Optimizer, And State ⭐⭐⭐

**Task:** provide thin Python wrappers for `Module`, `Linear`, ReLU, MSE,
cross-entropy, and SGD. `parameters()`/`state_dict()` names and identities must
match C++ semantics, and optimizer updates must remain native/no-grad operations.
Construction of `Module`, `Sequential`, and SGD must reject parameters from
different native contexts before a training step can partially execute.

For the first API, Python subclassing of native `Module` is optional. A simple
composition API is sufficient:

```python
model = comtam.nn.Sequential(
    comtam.nn.Linear(ctx, in_features, hidden),
    comtam.nn.ReLU(),
    comtam.nn.Linear(ctx, hidden, classes),
)
```

If `Sequential` is not implemented, construct the same model explicitly. Do not
add a generic hook/callback system merely to imitate a larger framework.

## Assignment 14.6: Build Python Conformance Tests ⭐⭐⭐

**Task:** mirror the C++ supported-surface matrix in pytest. Compare forward
results to NumPy, use numerical gradients for Python autograd entry points, and
run the deterministic regression plus a bounded MNIST smoke/training fixture.

Required categories:

- construction and NumPy round trip
- scalar/empty/view/broadcast/reduction/matmul semantics
- autograd identity, accumulation, `detach`, and `no_grad`
- module parameter names and SGD identity preservation
- mixed-context module/optimizer construction rejection
- snapshot save/load from Python and cross-language compatibility with C++
- lifetime after dropping the Python `Context`, and exception behavior

Tests must prove a checkpoint written in C++ loads in Python and vice versa.

## Assignment 14.7: Publish Type Information And API Docs ⭐⭐

**Task:** ship `.pyi` stubs or equivalent typed signatures for the supported API.
Document context ownership, synchronization, copy semantics, supported dtype/rank,
and one complete training example.

Generated stubs are acceptable only if reviewed against runtime behavior.

## Module 14 Checklist

- [ ] 14.1 Private extension and public package import in a clean environment.
- [ ] 14.2 Context/tensor/view/autograd identity follows native semantics.
- [ ] 14.3 NumPy interop uses explicit owning copies and synchronization.
- [ ] 14.4 Exceptions, GIL release, and in-flight lifetime are tested.
- [ ] 14.5 `nn`, optimizer, and state wrappers delegate to native code.
- [ ] 14.6 Pytest mirrors C++ conformance and cross-language snapshots.
- [ ] 14.7 Type information and runnable training documentation are shipped.

## Exit Criteria

You are ready for Module 15 when:

1. A clean environment imports the extension and runs a tensor operation.
2. Python tensors preserve native context, storage, and autograd identity.
3. NumPy interop has explicit owning-copy semantics and correct synchronization.
4. Exceptions and GIL/lifetime behavior are tested.
5. Python nn/optimizer/state APIs delegate to the native implementation.
6. Python and C++ conformance suites agree, including cross-language snapshots.
7. The supported Python API has accurate type information and examples.
