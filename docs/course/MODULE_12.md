## Module 12: Persistence And Reproducibility

Save model state in a versioned format, reload it safely, and resume the same
training workload.

## Why Persistence Is A Framework Feature

A checkpoint is more than a dump of bytes. It binds parameter names, dtypes,
shapes, byte order, format version, validation, and model ownership. Once users
store data, changing any of those rules becomes a compatibility problem, so the
format comes after the tensor and module contracts are stable.

## Module Contract

**Prerequisites:** Modules 10 and 11 pass, named parameters are stable, and host
readback has a defined synchronization/error boundary.

**You will produce:** deterministic `state_dict` enumeration, a versioned tensor
snapshot format, strict load validation, checkpoint round-trip and corruption
tests, and a save/load/resume training proof.

**Supported scope:** model parameters and optional optimizer state for the SGD
implemented in Module 7. Files are local and trusted only after validation.

**Not required:** arbitrary object serialization, executable graphs, pickle,
remote storage, compression, encryption, or compatibility with PyTorch formats.

## Assignment 12.1: Define `state_dict` ⭐⭐

**Task:** expose a deterministic mapping from hierarchical parameter names to
tensor handles. Loading copies values into existing parameter identities; it
must not replace registered parameters.

Define strict behavior for missing keys, unexpected keys, duplicate keys, dtype
mismatch, and shape mismatch. Start with strict loading; permissive loading may
be added later as an explicit option.

## Assignment 12.2: Specify The Snapshot Format Before Coding ⭐⭐⭐

**Task:** write a binary format note containing at least:

- magic bytes and format version
- byte order and integer widths
- tensor count and bounded name lengths
- per-tensor name, dtype, rank, dimensions, payload byte count, and checksum
- maximum accepted rank/dimensions/file size
- duplicate-name rejection
- truncation/trailing-data policy

Use fixed-width fields and checked arithmetic. Never allocate based on an
unvalidated count or multiply dimensions without overflow checks.

## Assignment 12.3: Implement Save And Strict Load ⭐⭐⭐

**Task:** save tensors in deterministic name order and load through a parser that
validates the complete file before mutating a live model.

Write to a sibling temporary file, flush/close successfully, then atomically
replace the destination where the platform supports it. A failed save must not
destroy the previous valid checkpoint.

**Tests:** empty state, multiple tensors, non-contiguous parameter policy,
truncation at representative offsets, bad magic/version/checksum, duplicate
names, oversized metadata, overflowed shapes, strict key mismatch, and unchanged
destination/model after failure.

## Assignment 12.4: Resume Training ⭐⭐⭐

**Task:** train the deterministic Module 7 model for `N` steps, save, reconstruct
a fresh model/context, load, and continue for `M` steps. Compare against an
uninterrupted `N+M` run.

If exact continuation requires RNG or optimizer state, include and version that
state explicitly. Do not claim resumability when only weights were restored.

## Assignment 12.5: Add Compatibility Fixtures ⭐⭐

**Task:** keep at least one small snapshot fixture from each released format
version and test loading it. The writer only emits the newest version; the reader
states exactly which older versions remain supported.

## Module 12 Checklist

- [ ] 12.1 Deterministic named `state_dict` and strict load semantics.
- [ ] 12.2 Bounded, versioned binary format written before implementation.
- [ ] 12.3 Atomic save and validate-before-mutate load with corruption tests.
- [ ] 12.4 Interrupted and resumed training matches the stated policy.
- [ ] 12.5 Compatibility fixtures pin every supported reader version.

## Exit Criteria

You are ready for Module 13 when:

1. `state_dict` names and load mutation semantics are deterministic.
2. The format is documented, bounded, versioned, and validated before allocation
   or model mutation.
3. Round-trip values, shapes, dtypes, and names match exactly.
4. Corrupt and adversarial metadata fail safely with the live model unchanged.
5. A saved run resumes according to the stated reproducibility policy.
6. Compatibility fixtures pin the reader's version promise.
