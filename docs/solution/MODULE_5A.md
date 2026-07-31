# Module 5A: Primitive Surface Consolidation

## Status

Blocked until the Module 5 exit gate passes. Do not implement or grade this
module against an incomplete broadcast/reduction/matmul baseline.

## Working Checklist

- [ ] 5A.1 Record the public, semantic, and physical operator inventories.
- [ ] 5A.2 Add strided unary `neg` and `recip` with independent oracles.
- [ ] 5A.3 Centralize metadata preflight and define the raw broadcast boundary.
- [ ] 5A.4 Compose `sub`, `div`, full `mean`, and axis `mean`; retire redundant and speculative dispatch paths.
- [ ] 5A.5 Define and test finite and exceptional numerical semantics.
- [ ] 5A.6 Clean-build and audit code, tests, documentation, and the correctness map.

## Design Decisions

Complete the operator inventory from Assignment 5A.1 here before editing
dispatch code. Record validation ownership, the internal raw-broadcast policy,
unary command-path design, numerical semantics, and any deviation from the
course recommendation with evidence.

## Completion Evidence

```text
configured     -> not run
compiled       -> not run
unit tested    -> not run
integrated     -> not run
gate passed    -> no; blocked by Module 5
not run        -> all Module 5A implementation checks
```

Record exact commands, focused test names, oracle/tolerance choices, clean-build
evidence, and a numbered mapping to every Module 5A exit criterion. Do not
replace this section with “works.”

## Agent Feedback / Grading

No grading yet. The module is intentionally blocked until Module 5 passes.
