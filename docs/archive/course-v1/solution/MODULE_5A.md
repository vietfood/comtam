# Module 5A: Primitive Surface Consolidation

## Assignment 5A.1: Write The Operator Inventory

### Agent Feedback / Grading

Status: passed.

The source and revised course now agree on six initially differentiable arithmetic primitives plus forward-only `max`. `min` is composed from `neg` and `max`; Module 6 must reject gradient recording through `max` or `min` until a tie policy and gradient test are deliberately added.

Operator inventory:

| Public API | Semantic primitive or composition | Physical entry point(s) | Validation owner | Forward oracle | Future backward owner |
| --- | --- | --- | --- | --- | --- |
| `add` | primitive | `add_fp32` | `check_binary` | MLX-C broadcast/scalar/layout tests | Module 6 add rule |
| `sub` | `add(a, neg(b))` | `neg_fp32`, `add_fp32` | public `sub` then primitive checks | MLX-C public binary tests | composition only |
| `mul` | primitive | `mul_fp32` | `check_binary` | MLX-C broadcast/scalar tests | Module 6 mul rule |
| `div` | `mul(a, recip(b))` | `recip_fp32`, `mul_fp32` | public `div` then primitive checks | MLX-C plus numerical classification | composition only |
| `neg` | primitive | `neg_fp32` | `check_unary` | MLX-C plus numerical classification | Module 6 neg rule |
| `recip` | primitive | `recip_fp32` | `check_unary` | MLX-C plus numerical classification | Module 6 reciprocal rule |
| `sum` | primitive | `reduce_sum_full_fp32`, `reduce_sum_axis_fp32` | reduction preflight | MLX-C reduction tests | Module 6 sum rule |
| `mean` | `mul(sum(a), scalar(1.0F / count))` | sum and multiply entry points | public mean reduction preflight | MLX-C full/axis tests | composition only; constant scale is non-recording |
| `max` | forward-only primitive | `reduce_max_full_fp32`, `reduce_max_axis_fp32` | reduction preflight | MLX-C full/axis tests | deferred; recording must reject until tie policy exists |
| `min` | `neg(max(neg(a)))` | negation and max entry points | public min reduction preflight | MLX-C full/axis tests | inherits forward-only max policy |
| `matmul` | primitive | `matmul_contiguous_fp32`, `matmul_strided_fp32` | `check_matmul` | MLX-C contiguous/strided tests | Module 6 matmul rule |
| `transpose` | public view op | none | `view::transpose` | MLX-C movement tests | Module 6 transpose rule |
| `permute` | public view op | none | `view::permute` | MLX-C movement tests | Module 6 permute rule |
| `shrink` | public view op | none | `view::shrink` | MLX-C movement tests | Module 6 shrink rule |
| `expand` | public view op | none | `view::expand` | MLX-C movement tests | Module 6 expand rule |
| `reshape` | public view op | none | `view::reshape` | MLX-C movement tests | Module 6 reshape rule |

Physical layout and reduction variants do not increase the semantic primitive count. Later internal backward-support kernels are allowed without becoming public differentiable primitives.

## Assignment 5A.2: Add Unary `neg` And `recip`

### Agent Feedback / Grading

Status: passed.

What is complete:

- `Op::NEG` and `Op::RECIP` have a narrow unary dispatch path.
- Both Metal kernels read through `physical_offset` and write contiguous output.
- Ordinary finite values pass MLX-C comparisons.
- Scalar IEEE cases for signed zero, infinity, and NaN pass.
- Unary metadata preflight rejects unsupported dtype, rank, and extents before output allocation.

Both unary primitives now pass MLX-C comparisons on a transpose and a nonzero-offset shrink. Public rank-5 and zero-extent rejection covers both `neg` and `recip`. Scalar exceptional-value classification supplies the rank-0 evidence.

## Assignment 5A.3: Make Validation Precede Composition

### Agent Feedback / Grading

Status: passed for architecture; remaining public negative cases are part of the shared test-completion work.

The `checks` helpers are side-effect-free and cover dtype support, rank four, positive extents, broadcasting, reduction axes, and matmul contracts. Public `sub`, `div`, and `mean` call preflight before launching their first internal primitive, which fixes the important eager-ordering bug.

The future tape boundary is now explicit. `tensor::bop` expands each operand's raw `view` and creates an internal storage-sharing tensor alias without calling public `tensor::expand`. User-authored `tensor::expand` remains the only expand operation visible to the future tape.

Public rank-5 and zero-extent cases now cover composed `div`. Because float32 is the only constructible dtype, direct `check_binary` unit coverage is accepted for dtype mismatch rather than manufacturing an invalid public tensor. The scalar test that throws during non-float host construction should be named as a constructor/dtype test, not as division-by-zero evidence.

## Assignment 5A.4: Compose `sub`, `div`, And `mean`

### Agent Feedback / Grading

Status: passed.

What is complete:

- `sub(a, b)` is `add(a, neg(b))` after binary preflight.
- `div(a, b)` is `mul(a, recip(b))` after binary preflight.
- `Op::SUB`, `Op::DIV`, their mappings, and their Metal entry points are gone.
- Full and axis `mean` are public compositions with rank-0 constants and no mean kernel.
- Public scalar/asymmetric broadcast tests cover all four binary names, and full/axis means pass MLX-C.

Full and axis mean now use the direct `mul(sum(...), scalar(1.0F / count))` composition. The constant scale is non-recording. All four public binary names pass the non-contiguous broadcast oracle. `mean` inherits the current contiguous-only `sum` layout contract; Module 5A does not add strided reductions. `max` is retained as a tested forward-only semantic primitive, while `min` is composed; their autograd behavior is explicitly deferred.

## Assignment 5A.5: Define Numerical Semantics

### Agent Feedback / Grading

Status: mostly passed.

The numerical suite is strong: it checks negation of signed zero, infinities, and NaN; reciprocal edges; division classification; and one fixed `1e-5` absolute-plus-relative tolerance for finite reciprocal and composed division against MLX-C. The current Metal command uses neither `-ffast-math` nor `-fno-signed-zeros`, and the runtime classification tests pass on this toolchain. The complete numerical contract, tolerance ownership, compiler policy, and currently unspecified behavior are recorded in `docs/ARCHITECTURE.md`.

Finish the documented sign-XOR matrix with representative negative denominators and signed infinities for zero/finite, zero/infinity, infinity/finite, and infinity/infinity. Also record the compiler command and numerical policy in this solution note. The test named `Scalar binary ops reject mismatched dtype and divide-by-zero` does not test division by zero: `tensor(0.0, ..., Float32)` throws on the host `double`/float32 mismatch before `tensor::div` runs. Rename that test to describe dtype rejection; the dedicated numerical suite already owns division-by-zero semantics.

## Assignment 5A.6: Perform The Static And Clean-Build Audit

### Agent Feedback / Grading

Status: passed for documentation and static source consistency.

Source inspection confirms there are no `Op::SUB`, `Op::DIV`, `sub_fp32`, or `div_fp32` dispatch artifacts in the current implementation. Before the latest internal broadcast-alias edit, an independently run incremental build and the 43-test Metal-backed suite passed; the user reports building the current revision successfully.

`docs/ARCHITECTURE.md` now matches the current primitive/composed surface, physical kernel variants, validation boundaries, forward-only `max` policy, internal broadcast aliases, and numerical contract. The current revision's build and tests are user-verified and were not independently rerun, per the user's explicit instruction.

## Module 5A Verdict

Status: not passed yet.

The operator inventory, internal broadcast tape boundary, `MAX` policy, mean composition, strided operation coverage, and architecture documentation are resolved. Module 5A remains open only for two small numerical/validation evidence corrections.

Prioritized remaining work:

1. Add representative negative-denominator/signed-infinity division cases so every documented sign-XOR row is executable evidence.
2. Rename the misleading scalar dtype/divide-by-zero test; the dedicated numerical suite owns division-by-zero behavior.

Verification:

```text
compiled        -> user-verified on current revision; not independently rerun
full CTest      -> user-verified on current revision; exact count/command not supplied
static audit    -> passed
gate passed     -> no
```

Commands independently run before the latest alias edit:

```text
cmake --build build -j 6
ctest --test-dir build --output-on-failure
xcrun -sdk macosx metal --help | rg -n "fast-math|finite-math|honor|signed-zero|denormal|NaN|Inf"
```

Do not start Module 6 yet. Finish the semantic boundary now so autograd records one intentional graph rather than inheriting backend aliases and unresolved primitives.
