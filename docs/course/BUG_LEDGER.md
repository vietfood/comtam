# Bug Ledger

This ledger is for concrete bugs and gaps discovered while running the course.

comtam is young, so many entries here will be "not implemented yet" rather than
"wrong". Classify them honestly. A bug belongs here only after you have a repro,
an oracle, and a first guess about which module owns it.

## Rules

- Minimize the repro before writing the entry.
- Record the oracle: manual, CPU loop, or traceback only.
- Classify the owning module.
- Fix at most one bug per patch.
- Do not mix bug fixes with cleanup.
- If the behavior is unimplemented rather than wrong, mark it unsupported.

## Entry Template

```text
ID:
Title:
Status:
  - open
  - fixed
  - documented unsupported
  - deferred
Minimal repro:
Oracle:
  - manual expected value
  - one-off CPU comparison
  - traceback only
Expected:
Actual:
Traceback or wrong output:
First suspicious file/function:
Concept owner:
  - Module 0 codebase/Metal context
  - Module 1 storage/ownership
  - Module 2 views/indexing
  - Module 3 eager dispatch
  - Module 4 forward correctness
  - Module 5 broadcast/reduce/matmul
  - Module 6 autograd
  - Module 7 nn/optim
  - Module 8 training
  - Module 9 hardening/perf
Severity:
  - blocks tracing
  - wrong result
  - unsupported feature
  - confusing design
Next action:
  - fix now because it blocks progress
  - convert to failing test in later module
  - document as unsupported
Notes:
```

## Open Bugs

Add entries below this line.
