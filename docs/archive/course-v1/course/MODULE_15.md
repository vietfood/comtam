## Module 15: Distribution And Release Engineering

Make the narrow supported runtime reproducibly buildable, installable, testable,
and diagnosable by someone who did not develop it.

## Why This Is The Production Gate

Correct source code in one working tree is not a product boundary. A production-
quality narrow framework needs supported environments, installable artifacts,
automated conformance, compatibility fixtures, performance history, failure
diagnostics, and a repeatable release process.

This gate does not claim PyTorch-scale production maturity. It claims that the
documented Apple Silicon/Metal/float32 workload is supported with evidence.

## Module Contract

**Prerequisites:** Modules 10-14 pass, including the Python API and cross-language
snapshot compatibility.

**You will produce:** installable CMake packages, Apple Silicon Python wheels,
CI/release workflows on supported hardware, sanitizer and conformance jobs,
versioned documentation/changelog, release artifacts, and a clean-machine
acceptance run.

**Supported scope:** explicitly listed macOS, Xcode/Metal toolchain, Apple
Silicon families, C++ compiler, CMake, and CPython versions.

**Not required:** Linux/Windows, Intel Mac, CUDA, source-compatible ABI forever,
PyPI publication before the project is ready, or availability guarantees beyond
the written support policy.

## Assignment 15.1: Install The Native Package ⭐⭐⭐

**Task:** provide namespaced CMake targets and install rules for public headers,
native libraries, and required Metal library resources.

A downstream project must succeed with the equivalent of:

```cmake
find_package(comtam CONFIG REQUIRED)
target_link_libraries(app PRIVATE comtam::comtam)
```

Test both build-tree and install-tree consumption. The installed package must not
refer back to source/build absolute paths.

## Assignment 15.2: Build Reproducible Python Wheels ⭐⭐⭐

**Task:** build platform-tagged Apple Silicon wheels containing the extension,
Python package, type information, licenses, and Metal resources. Define how the
extension finds its `.metallib` after installation.

Test each supported CPython version in a fresh virtual environment using the
built wheel, not an editable source checkout.

## Assignment 15.3: Define The Support Matrix And CI ⭐⭐⭐

**Task:** write the exact supported combinations and run CI on real Apple/Metal
hardware for them.

CI stages should distinguish:

- formatting/static host checks
- C++ build and CTest
- Python wheel build and pytest-from-wheel
- gradient/conformance/persistence fixtures
- bounded training and async stress
- sanitizer jobs where compatible
- scheduled or dedicated performance tracking

A job skipped because no Metal device exists is not a passing Metal test. Report
the environmental gap explicitly.

## Assignment 15.4: Define Compatibility And Security Handling ⭐⭐

**Task:** document API/version compatibility, snapshot reader support, dependency
update policy, vulnerability reporting, and malformed-file threat boundaries.

Run dependency/license inventory and ensure release archives contain required
notices. Persistence fuzzing or structured malformed-input generation should run
against the snapshot parser because it accepts untrusted lengths and shapes.

## Assignment 15.5: Write User Documentation ⭐⭐

**Task:** write documentation for installation, supported environments, context
ownership, tensor semantics, autograd, training, checkpointing, synchronization,
known limitations, and troubleshooting Metal/toolchain failures.

Every example must run in CI. Avoid documenting planned APIs as if released.

## Assignment 15.6: Rehearse The Release ⭐⭐⭐

**Task:** create a release candidate from a clean tagged commit, build native and
Python artifacts, verify checksums, install them on a clean machine/environment,
and run one end-to-end acceptance scenario:

```text
install
  -> train or load the supported model
  -> save checkpoint
  -> restart in a fresh process
  -> load and run inference
  -> compare expected output/accuracy
  -> run benchmark smoke and report version/build information
```

Do not publish externally merely to satisfy the course. A local release rehearsal
with preserved artifacts and logs proves the pipeline first.

## Assignment 15.7: Make A Production Claim With Boundaries ⭐

**Task:** write a one-page readiness statement containing:

- the exact workload and environments supported
- correctness, training, stress, compatibility, and performance evidence
- known limitations and failure modes
- which guarantees are best-effort versus release-blocking
- the next condition that would justify broadening scope

Avoid calling comtam a general-purpose production framework. The defensible claim
is a production-quality runtime for the narrow contract you have actually tested.

## Module 15 Checklist

- [ ] 15.1 Native install/build-tree consumers use `comtam::comtam`.
- [ ] 15.2 Supported CPython versions install and test built wheels.
- [ ] 15.3 Real Apple/Metal CI runs the declared support matrix.
- [ ] 15.4 Compatibility, malformed input, dependencies, and licenses are checked.
- [ ] 15.5 Published examples run against packaged artifacts.
- [ ] 15.6 A clean release-candidate rehearsal preserves artifacts and logs.
- [ ] 15.7 The readiness statement limits claims to measured support.

## Exit Criteria

The production track is complete when:

1. External C++ and Python consumers install only released artifacts and pass
   their acceptance tests.
2. Supported Apple hardware runs native, Python, persistence, training, lifetime,
   and performance jobs automatically.
3. Snapshot/API compatibility and malformed-input handling are release-tested.
4. Documentation examples execute against the packaged artifacts.
5. A clean release rehearsal produces traceable artifacts, checksums, logs, and
   version information.
6. The readiness statement makes a narrow claim backed by the recorded evidence.
