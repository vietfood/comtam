## Module 10: Supported Tensor Semantics

Turn the Module 9 behavior matrix into a stable public tensor contract suitable
for code outside this repository.

## Why This Starts The Production Track

Production quality begins with predictability. Users can work around a missing
operator, but they cannot safely work around ambiguous aliasing, mutation,
contiguity, axis, and error rules. This module stabilizes those semantics before
async execution, files, or language bindings make them expensive to change.

## Module Contract

**Prerequisites:** all Module 9 gates pass and the v0 supported-semantics matrix
exists.

**You will produce:** a reviewed public API boundary, stable tensor identity and
alias rules, storage versioning for mutation safety, a materialization operation,
normalized axis semantics, and conformance tests for every supported operation.

**Supported scope:** the existing Metal/float32/eager runtime. Add a dtype or op
only when a selected workload requires it and you can define its full contract.

**Not required:** Python convenience, multiple devices/backends, NumPy-style
dtype promotion, or compatibility with every PyTorch edge case.

**Completion evidence:** an API/semantics document, a conformance test manifest,
and compatibility notes for any change from the core-course API.

## Assignment 10.1: Define Public And Internal Headers ⭐⭐

**Task:** Decide which headers and symbols an external C++ user may depend on.
Keep Metal implementation details, kernel descriptors, and internal autograd
nodes out of that surface.

The public API must make context ownership visible and use one naming/error
policy consistently. Add a minimal external-consumer CMake target that includes
only public headers and links the library without reaching into source paths.

**Test:** build a small consumer that creates a context, constructs tensors,
runs a differentiable expression, and reads the result using public APIs only.

## Assignment 10.2: Define Aliasing And Mutation ⭐⭐⭐

**Task:** Document which operations share storage, which allocate, and what
mutation means for tensor handles and saved autograd values.

Add a monotonically increasing storage version. Every supported mutation bumps
it; a backward node that saved values also saves their versions and rejects a
changed input. Define whether host upload, parameter update, and future in-place
ops count as mutation. They normally should.

**Tests:** alias visibility, non-aliasing allocation, version increments,
mutation-before-backward rejection, and safe optimizer mutation after a consumed
graph.

## Assignment 10.3: Add `contiguous` Materialization ⭐⭐⭐

**Task:** Add one explicit operation that returns the same logical values in a
contiguous allocation. If the input is already contiguous, choose and document
whether it returns an alias or a copy.

Use a GPU gather kernel for supported views rather than making CPU readback part
of device execution. This becomes the escape hatch for kernels that deliberately
support only contiguous inputs.

**Tests:** transpose, shrink with non-zero offset, expand with stride zero,
already-contiguous input, scalar, empty tensor, and backward mapping.

## Assignment 10.4: Normalize Shape And Axis Semantics ⭐⭐

**Task:** Centralize shape/axis validation rather than letting each op interpret
axes differently.

Decide and test:

- whether negative axes are accepted; if accepted, normalize exactly once
- whether multiple axes are sorted or preserve caller order
- duplicate-axis behavior
- rank and dimension-size limits
- `keepdim` behavior for every reduction
- scalar-axis behavior

The recommended policy is Python-style negative single axes, duplicate rejection,
and one shared normalization helper.

## Assignment 10.5: Build The Conformance Matrix ⭐⭐⭐

**Task:** Generate or hand-maintain parameterized tests that apply the Module 9
matrix to every public op. Each row identifies forward oracle, gradient oracle,
supported layouts, edge shapes, and expected errors.

The test manifest is part of the supported API. A new op is incomplete until it
has a row and the corresponding tests.

## Assignment 10.6: Version The API Contract ⭐

**Task:** assign an initial semantic version to the public API and state what
counts as a breaking change before 1.0. Record changes in a changelog from this
point onward.

## Module 10 Checklist

- [ ] 10.1 External consumers use public headers and a supported target only.
- [ ] 10.2 Aliasing, mutation, and storage versions are documented and tested.
- [ ] 10.3 Device-side `contiguous` covers every supported view form.
- [ ] 10.4 One helper owns shape and axis normalization rules.
- [ ] 10.5 The conformance matrix covers every public operation.
- [ ] 10.6 API version and pre-1.0 compatibility policy are recorded.

## Exit Criteria

You are ready for Module 11 when:

1. An external C++ consumer builds without internal headers.
2. Alias, copy, mutation, and autograd-version behavior are explicit and tested.
3. Any supported view can be materialized contiguously on the device.
4. Shape and axis rules are centralized across public ops.
5. The conformance matrix covers the complete supported surface.
6. API evolution has a written versioning policy.
