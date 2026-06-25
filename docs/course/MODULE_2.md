## Module 2: Views And Strides

The `View` is the heart of zero-copy tensor operations.

If you understand views, you understand why `transpose`, `reshape`, `expand`,
`slice`, and many gradient rules do not move data. If you do not understand
views, every later kernel will feel haunted.

This module goes slowly on purpose. It is cheaper to be slow here than to debug
wrong kernels later.

## Why This Module Exists

Views are the foundation of the whole framework.

Every later module assumes that a logical tensor index can be lowered into a
physical storage offset. Dispatch, broadcasting, reductions, matmul layout, and
gradient rules all depend on that mapping being correct.

The main design decision in this module is to put indexing knowledge in `View`,
not scattered across `Tensor`, `Device::copy`, and your Metal kernels.

The current `View` already stores the right fields (`comtam/core/view.h`):

```cpp
struct View {
    std::vector<ViewInt> shape;
    std::vector<ViewInt> strides;
    std::vector<ViewInt> ref_strides;  // contiguous reference, for is_contiguous
    size_t offset = 0;
};
```

But it only knows how to build a contiguous view. It cannot yet transpose,
slice, expand, or compute a physical offset from a logical index. That is this
module's work.

## Mental Model

A view answers one question:

```text
Given a logical index, which element of the underlying storage do I read?
```

For a view:

```text
shape   = (2, 3)
strides = (3, 1)
offset  = 0
```

the mapping is:

```text
physical_offset = offset + i0 * stride0 + i1 * stride1
```

So:

```text
logical (0, 0) -> 0 + 0*3 + 0*1 = 0
logical (0, 1) -> 0 + 0*3 + 1*1 = 1
logical (1, 0) -> 0 + 1*3 + 0*1 = 3
logical (1, 2) -> 0 + 1*3 + 2*1 = 5
```

The storage is still one flat `MTL::Buffer`. Shape and strides are the
interpretation.

## Core Vocabulary

`shape`: the logical dimensions the user sees, e.g. `(2, 3)`.

`strides`: how far to move in physical storage when a logical dimension
increases by one. For row-major contiguous `(2, 3)`, `strides == (3, 1)`. Moving
one row jumps 3 elements; moving one column jumps 1.

`ref_strides`: the strides a contiguous tensor of this shape would have. comtam
keeps this so `is_contiguous()` is a simple comparison (`view.h:25`). Most
frameworks recompute this instead of storing it. Either is fine; know the
tradeoff (stored is faster to check, but must be kept in sync).

`offset`: where the view starts inside the storage. Slicing changes offset.

`linear_idx`: the flat output index a kernel iterates over, `0 .. numel()-1`.

`physical_offset`: the actual storage position after applying strides and offset.

## Linear Index Decomposition

A Metal kernel receives a flat thread id. For a logical shape `(2, 3, 4)`,
convert a linear index into coordinates by peeling dimensions from the right:

```text
i2 = linear_idx % 4
tmp = linear_idx / 4
i1 = tmp % 3
tmp = tmp / 3
i0 = tmp % 2
```

Then apply strides:

```text
physical_offset = offset + i0*stride0 + i1*stride1 + i2*stride2
```

This is the key formula for the whole module. It works for contiguous,
transposed, broadcasted, and sliced views without any special cases.

## Worked Example 1: Contiguous

```text
shape   = (2, 3)
strides = (3, 1)
offset  = 0
linear_idx = 4
```

Decompose: `col = 4 % 3 = 1`, `tmp = 4 / 3 = 1`, `row = 1 % 2 = 1`.

Apply: `0 + 1*3 + 1*1 = 4`.

For a contiguous view, this always equals `linear_idx + offset`.

## Worked Example 2: Transpose

Start contiguous `(2, 3)` with `strides = (3, 1)`. Transpose to:

```text
shape   = (3, 2)
strides = (1, 3)
```

No data moved. Only the interpretation changed. For `linear_idx = 4` in the
transposed shape `(3, 2)`: `col = 0`, `row = 2`, so
`physical_offset = 0 + 2*1 + 0*3 = 2`. That is correct, because transposed
logical `(2, 0)` is original storage `(0, 2)`.

## Worked Example 3: Broadcast

Start `(3, 1)` with `strides = (1, 0)` conceptually, expand to `(3, 4)`:

```text
shape   = (3, 4)
strides = (1, 0)
```

The second dimension has stride `0`, so every column reads the same element.
This is why `expand` is zero-copy. For `linear_idx = 5`: `col = 1`, `row = 1`,
`physical_offset = 1*1 + 1*0 = 1`.

## Worked Example 4: Slice

```text
x.shape = (5,), x.strides = (1,), x.offset = 0
y = x[2:5]
y.shape = (3,), y.strides = (1,), y.offset = 2
```

`y[0] -> 2 + 0*1 = 2`, `y[2] -> 2 + 2*1 = 4`. No data moved; only offset changed.

## Worked Example 5: The Contiguous Shortcut

For a contiguous view, the decomposition collapses to:

```text
physical_offset = linear_idx + offset
```

So a kernel over a contiguous tensor never needs modulo or divide. This is the
first real optimization, and it is safe because it is exact, not approximate.
comtam already exposes `is_contiguous()` (`view.h:25`); this module makes it pay
off.

## Assignment 2.1: Verify Contiguous Stride Construction ⭐

**Task:** The current `View(shape)` constructor (`view.h:17`) computes strides
right-to-left. Verify it by hand and by test.

Questions:

1. For `shape = (2, 3, 4)`, what are `strides` and `ref_strides`?
2. Why does the loop run from `size()-2` down to `0`?
3. What does the constructor produce for a scalar `shape = ()` and for a
   zero-size dimension like `(0, 3)`? Are those correct or undefined?

**Test to write:** Assert strides for several shapes, including a 1-D, a 3-D, and
a degenerate shape. Decide explicitly what a scalar view should be.

**Why this assignment:** Every other view op starts from a correct contiguous
view. If construction is wrong, nothing downstream can be trusted.

## Assignment 2.2: Implement `physical_offset(linear_idx)` ⭐⭐

**Task:** Add one source of truth for logical-to-physical mapping on `View`:

```cpp
// returns offset + sum(coord[d] * strides[d]) for a flat linear index
size_t physical_offset(size_t linear_idx) const;
```

Rules:

1. Use `shape`, `strides`, and `offset`.
2. Decompose the linear index from the right, exactly as in the worked examples.
3. Do not special-case transpose or broadcast. The formula already covers them.
4. Add a contiguous fast path using `is_contiguous()`.

**Tests to write:**

```text
contiguous (2,3): physical_offset(4) == 4
transpose (3,2) strides (1,3): physical_offset(4) == 2
broadcast (3,4) strides (1,0): physical_offset(5) == 1
slice offset=2 (3,) strides (1,): physical_offset(0) == 2, (2) == 4
```

**Study prompt:** Why does the decomposition loop run over dimensions in reverse
order?

## Assignment 2.3: Implement `permute` And `transpose` ⭐⭐

**Task:** Add view ops that reorder axes without moving data.

```cpp
View permute(const std::vector<size_t>& axes) const;  // general reorder
View transpose(size_t a, size_t b) const;             // swap two axes
```

Rules:

- Reorder `shape` and `strides` together by `axes`.
- Do not touch `offset`.
- After a permute, `is_contiguous()` should usually return false. Confirm that
  `ref_strides` still reflects the contiguous layout of the new shape, or decide
  to recompute it. Write down which you chose and why.

Questions:

1. Is `transpose(a, b)` just `permute` with two indices swapped?
2. What is the inverse of a permute, and why will Module 6 need it?

**Test:** Build a `(2, 3)`, transpose it, read it back element by element via
`physical_offset`, and compare to a CPU transpose.

## Assignment 2.4: Implement `slice` / `shrink` ⭐⭐

**Task:** Crop a tensor to absolute bounds.

```cpp
View shrink(const std::vector<std::pair<ViewInt, ViewInt>>& limits) const;
```

View rule:

```text
new_shape[i]  = end_i - start_i
new_offset    = old_offset + sum(start_i * strides[i])
new_strides   = old_strides
```

Questions:

1. Is `shrink` just `slice` with a clearer name?
2. How should out-of-range bounds behave? Throw, or clamp?
3. Should negative bounds be supported?

**Recommended first choice:** Reject negative and out-of-range bounds with a
clear error. Keep it absolute and boring. Do not add Python-style negative
indexing yet.

**Test:** `shrink` a `(4, 4)` to the inner `(2, 2)` and compare values against a
hand-computed expectation.

## Assignment 2.5: Implement `expand` (Broadcast) ⭐⭐

**Task:** Expand size-1 dimensions to a larger size using stride 0.

```cpp
View expand(const std::vector<ViewInt>& new_shape) const;
```

Rules:

- A dimension of size 1 can expand to any size by setting its stride to 0.
- A dimension whose size already matches stays unchanged.
- A dimension that is neither 1 nor equal is an error.

Questions:

1. Why does stride 0 make every index along that axis read the same element?
2. After expand, is the view contiguous? What does `to_vector` need to do
   differently for it? (This is the bridge to Assignment 2.7.)

**Test:** Expand `(3, 1)` to `(3, 4)` and verify each row repeats correctly.

## Assignment 2.6: Implement `reshape` And Its Failure Message ⭐⭐⭐

**Task:** Implement `reshape` for the cases that can be a pure view, and fail
loudly for the cases that cannot.

```cpp
View reshape(const std::vector<ViewInt>& new_shape) const;
```

Rules:

1. `numel()` must be preserved.
2. A contiguous view can always be reshaped by recomputing contiguous strides.
3. A non-contiguous view sometimes cannot be reshaped without copying.

Questions:

1. Can `(2,3,4).permute((2,0,1)).reshape((4,6))` be one view? If not, why?
2. When reshape cannot be a view, should comtam error or materialize with a
   `contiguous()` copy?

**Recommended first choice:** Error with a message that explains which stride
relationship broke. Add `contiguous()` (a real copy) as the escape hatch, but do
not silently copy inside `reshape`. Silent copies hide cost.

**Study prompt:** Write down one reshape-after-permute that can be a view, and
one that cannot.

## Assignment 2.7: Make Readback View-Aware ⭐⭐⭐

**Task:** Today `Tensor::to_vector` (`tensor.cpp:43`) `memcpy`s the whole buffer
and ignores strides and offset. That is correct only for contiguous, offset-0
views. Fix it.

Two valid designs:

- **A (CPU gather):** read the raw buffer, then gather on the CPU using
  `View::physical_offset(i)` for `i` in `0 .. numel()-1`. Simple, correct, slow.
- **B (materialize on GPU):** run a copy kernel that writes a contiguous result,
  then `memcpy`. Faster, but needs Module 3's dispatch first.

**Recommended first choice:** Implement A now. It makes every view op from this
module testable immediately, without waiting for kernels. Revisit B in Module 9
only if a measurement says readback dominates.

**Test:** Transpose a `(2, 3)`, call `to_vector`, and compare against a CPU
transpose of the original data, element for element.

**Why this assignment:** This is the assignment that turns views from "looks
right on paper" into "verified against an oracle". Until readback respects
strides, every view test is lying to you.

## Assignment 2.8: Movement Op Gradient Preview ⭐⭐

You will not implement these until Module 6, but understand the pairings now:

| Forward op | Backward op |
| --- | --- |
| `reshape` | `reshape` back |
| `permute` | inverse `permute` |
| `expand` | `sum` over expanded (stride-0) dimensions |
| `slice` / `shrink` | `pad` |

**Study prompt:** Why is the backward of `expand` a reduction? If element `a[i]`
was read four times via a stride-0 axis, how many gradient contributions must it
receive?

## Assignment 2.9: Read Magnetron / tinygrad Views ⭐⭐⭐

After your own attempt, read how a mature framework models views.

Questions:

1. What does it store in a view that comtam does not?
2. Does it stack views (a ShapeTracker) or keep one view per tensor?
3. How does it represent padding, where stride math alone is not enough?
4. Which idea would be overkill for comtam right now?

**Deliverable:** A short note titled "What comtam should copy from mature views,
and what it should postpone." comtam keeps one `View` per tensor for now; record
the first concrete case where that becomes painful.

## Module 2 Checklist

- [ ] 2.1 Verify contiguous stride construction with tests.
- [ ] 2.2 Implement `physical_offset(linear_idx)` with a contiguous fast path.
- [ ] 2.3 Implement `permute` and `transpose`.
- [ ] 2.4 Implement `shrink` / `slice`.
- [ ] 2.5 Implement `expand` with stride 0.
- [ ] 2.6 Implement `reshape` with a clear failure message.
- [ ] 2.7 Make `to_vector` view-aware (CPU gather).
- [ ] 2.8 Write the movement-op gradient preview note.
- [ ] 2.9 Read mature view code; write a copy/postpone note.

## Exit Criteria

You are ready for Module 3 when:

1. You can compute physical offsets by hand for contiguous, transposed,
   broadcasted, and sliced views.
2. `View::physical_offset` is the single source of truth for indexing.
3. `to_vector` respects strides and offset, verified against CPU oracles.
4. `reshape` fails loudly instead of silently copying.
5. You can explain why `expand` backward is a reduction.
