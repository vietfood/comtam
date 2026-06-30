## Assignment 2.1: Verify Contiguous Stride Construction

1. For `shape = (2, 3, 4)` the stride is `(12, 4, 1)`. It's really intuitive to compute stride, the right-most dimension is the fastest-varying dimension (only need `1` step). 
2. `size-1` is the right-most dimension and right-most dimension always has `stride = 1`. So we must run from the next element (`size-2`) to the first element (`0`).
3. If we set `shape = ()` the stride will be an empty vector too (but note that we should make the stride to be `(0)` so other code won't get `out_of_range` error). Now for the shape `(0, 3)`, the stride is actually `(0, 1)`. This is correct for the case `(0, 3)` but not for the case `()`.

### Agent Feedback / Grading

Status: mostly passed for Assignment 2.1. The core contiguous stride construction is correct and tested; only the degenerate edge cases still need tests.

What is good:

- `(2, 3, 4) -> (12, 4, 1)` is correct.
- The explanation of the right-most dimension being fastest-varying is the right mental model.
- The concise `tests/core/view.cpp` structure is good: each test has a small, obvious target.
- The implementation directly tests 1-D and 3-D contiguous strides and `ref_strides`.
- Treating scalar views as one contiguous element is the right rule for this project.

Remaining work:

- Add scalar behavior for `View({})`: `strides == {}`, `ref_strides == {}`, `numel() == 1`, and `is_contiguous() == true`.
- Add zero-size behavior for `View({0, 3})`: expected `strides == {3, 1}`, `ref_strides == {3, 1}`, and `numel() == 0`.
- Update the written answer for `(0, 3)`: current row-major construction gives `(3, 1)`, not `(0, 1)`. Zero elements and stride-zero broadcasting are different ideas.

## Assignment 2.2: Implement physical_offset(linear_idx)

```cpp
size_t physical_offset(size_t linear_index) const {
    // linear index shouldn't be larger than number of elements
    if (linear_index >= numel()) {
        throw std::invalid_argument("View::physical_offset: linear index is out of bounds");
    }

    if (shape.size() == 0) {
        return offset;
    }

    // if tensor is already contiguous
    if (is_contiguous()) {
        return linear_index + offset;
    }

    // tensor isn't contiguous
    std::vector<ViewInt> dec(shape.size());

    // we traverse in reverse
    // decompose linear index first
    size_t tmp = linear_index;
    for (size_t i = shape.size(); i-- > 0;) {
        dec[i] = tmp % shape[i];
        tmp /= shape[i];
    }

    // then calculate physical offset
    size_t result = offset;
    for (size_t i = 0; i < dec.size(); ++i) {
        result += dec[i] * strides[i];
    }

    return result;
}
```

### Agent Feedback / Grading

Status: passed for Assignment 2.2's required behavior. `View::physical_offset` is now the single source of truth for logical-to-physical indexing.

What is good:

- The real implementation in `comtam/core/view.h` uses `shape`, `strides`, and `offset` as the single source of truth.
- It has the required contiguous fast path.
- It passes tests for contiguous, transposed, broadcasted, and sliced views.
- The real code correctly uses `for (size_t i = shape.size(); i-- > 0;)` for reverse iteration over an unsigned index.
- Scalar `physical_offset` checks bounds before the rank-0 case and returns `offset` for `physical_offset(0)`, matching the "scalar is one contiguous element" rule.
- The test covers all four required examples from the assignment without involving Tensor or Metal.

Remaining work:

- Keep the current reverse-loop form: `for (size_t i = shape.size(); i-- > 0;)`. Do not change this to `for (size_t i = shape.size() - 1; i >= 0; --i)`, because that version is wrong for `size_t` and will underflow.
- Add tests for the scalar behavior you chose: `View({}, {}, offset).physical_offset(0) == offset` and `physical_offset(1)` throws.
- Add a zero-size bounds test. For `View({0, 3})`, every `physical_offset(i)` should throw before any division by `shape[i]`.

Verification is summarized after Assignment 2.3.

## Assignment 2.3: Implement permute And transpose

```cpp
/*
 * - If shape is (a, b, c) and new_axis is (0, 2, 1),
 * then the new shape is (a, c, b).
 * - Reference: https://pytorch.org/docs/stable/generated/torch.permute.html
 */
View permute(const ViewVector &new_axis) const {
    if (new_axis.size() != shape.size()) {
        throw std::invalid_argument("Permutation new axis doesn't match with current shape");
    }

    bool is_valid = std::all_of(new_axis.begin(), new_axis.end(), [&shape = shape](ViewInt axis) {
        return axis >= 0 && static_cast<size_t>(axis) < shape.size();
    });
    if (!is_valid) {
        throw std::invalid_argument("Invalid permute axis for current shape");
    }

    std::vector<bool> seen(shape.size(), false);
    for (ViewInt axis : new_axis) {
        auto idx = static_cast<size_t>(axis);
        if (seen[idx]) {
            throw std::invalid_argument("Permutation axes must be unique");
        }
        seen[idx] = true;
    }

    // create a new shape with permutation index
    ViewVector new_shape(shape.size());
    ViewVector new_strides(strides.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        new_shape[i] = shape[new_axis[i]];
        new_strides[i] = strides[new_axis[i]];
    }
    return {new_shape, new_strides, offset};
}

View transpose(ViewInt a, ViewInt b) const {
    if (a < 0 || b < 0 || static_cast<size_t>(a) >= shape.size() ||
        static_cast<size_t>(b) >= shape.size()) {
        throw std::invalid_argument("Transpose axes are out of bounds");
    }

    // build a full axis first
    ViewVector axes(shape.size());
    for (size_t i = 0; i < axes.size(); ++i) {
        axes[i] = static_cast<ViewInt>(i);
    }
    // then transpose the a and b from that axis
    std::swap(axes[static_cast<size_t>(a)], axes[static_cast<size_t>(b)]);

    return permute(axes);
}
```

### Agent Feedback / Grading

Status: passed for Assignment 2.3's core implementation and tests.

What is good:

- `permute` correctly validates rank, rejects out-of-range axes, and rejects duplicate axes.
- `permute` reorders `shape` and `strides` together and preserves `offset`.
- Returning `View(new_shape, new_strides, offset)` recomputes `ref_strides` for the new logical shape, which is the right choice for this design.
- `transpose(a, b)` now has the right semantics: it builds an identity axis list, swaps `a` and `b`, and delegates to `permute`.
- The tests cover multiple 3-D permutations, both `transpose(0, 1)` and `transpose(1, 0)` on a 2-D view, a 3-D transpose, and double-transpose round trips.
- The CPU value check is now at the right abstraction level: it gathers from a flat vector with `View::physical_offset` instead of pulling in `Tensor`, `Context`, or Metal.

Remaining work:

- The pure CPU view tests now use `[view]` plus behavior-specific tags instead
  of the misleading `[metal]` tag.
- Invalid `permute` and `transpose` inputs are now covered by tests.

Verification is summarized after Assignment 2.4.

## Assignment 2.4: Implement slice / shrink

```cpp
View shrink(const PairViewVector &limits) const {
    if (limits.size() != shape.size()) {
        throw std::invalid_argument("The numbers of each limit should match with shape size");
    }

    bool is_valid_limits = true;
    for (size_t i = 0; i < limits.size(); ++i) {
        is_valid_limits &= (limits[i].first < limits[i].second)               // start < end
                            && (limits[i].first >= 0 && limits[i].second >= 0) // non-negative
                            && (limits[i].second <= shape[i]); // end <= shape (in bound)
    }

    if (!is_valid_limits) {
        throw std::invalid_argument("limits is not valid");
    }

    ViewVector new_shape(shape.size());
    size_t new_offset = offset;
    for (size_t i = 0; i < shape.size(); ++i) {
        new_shape[i] = limits[i].second - limits[i].first;
        new_offset += limits[i].first * strides[i];
    }
    return {new_shape, strides, new_offset};
}
```

### Worked Example: Cropping A 4x4 View

Start with a contiguous 4x4 tensor:

```text
shape   = (4, 4)
strides = (4, 1)
offset  = 0

logical values:
[[ 0,  1,  2,  3],
 [ 4,  5,  6,  7],
 [ 8,  9, 10, 11],
 [12, 13, 14, 15]]
```

Now shrink to the inner 2x2 block:

```cpp
auto y = x.shrink({{1, 3}, {1, 3}});
```

Interpret each pair as `[start, end)` for one axis:

```text
axis 0: keep rows [1, 3) -> rows 1 and 2
axis 1: keep cols [1, 3) -> cols 1 and 2
```

The new view is:

```text
new_shape  = (3 - 1, 3 - 1) = (2, 2)
new_strides = old_strides = (4, 1)
new_offset = 0 + 1*4 + 1*1 = 5
```

No data moves. The view just starts at storage element 5 and keeps the old row
stride:

```text
y logical values:
[[ 5,  6],
 [ 9, 10]]
```

Check it through `physical_offset`:

```text
y[0,0] -> linear 0 -> physical offset 5
y[0,1] -> linear 1 -> physical offset 6
y[1,0] -> linear 2 -> physical offset 9
y[1,1] -> linear 3 -> physical offset 10
```

### Agent Feedback / Grading

Status: passed for Assignment 2.4's required behavior. `shrink` now crops by
adjusting shape plus offset, preserves the old strides, and has a focused value
test.

What is good:

- The implementation checks rank, rejects negative and out-of-range bounds, and
  keeps the operation absolute instead of adding Python-style indexing early.
- `new_offset += start_i * strides[i]` is exactly the right rule, including for
  non-contiguous parents.
- Returning `View(new_shape, strides, new_offset)` preserves old strides while
  recomputing `ref_strides` for the cropped logical shape, so `is_contiguous()`
  correctly becomes false for a cropped 2-D view whose row stride is still 4.
- The test covers the required 4x4 -> inner 2x2 example and verifies physical
  offsets `{5, 6, 9, 10}` against a hand-computed oracle.

Remaining work:

- Decide whether empty slices should be valid. Right now `start == end` throws,
  but Module 2 already asks you to reason about zero-size dimensions, so allowing
  `[k, k)` may be the more consistent rule.
- Negative `shrink` cases are now covered: wrong rank, negative start, end past
  shape, empty slice, and start greater than end.
- Clean up the comment in `View::shrink`: `InPython` needs a space, and examples
  like `{{1:2}, {1:max_shape}}` are not C++ syntax.

Verification:

- `cmake -S . -B build -DCOMTAM_BUILD_TESTS=ON` passed.
- `cmake --build build` passed.
- `ctest --test-dir build --output-on-failure -R "strides construction test|physical_offset test|permute and transpose|shrink"` passed: 4/4 focused view tests.
- `ctest --test-dir build --output-on-failure` failed inside the sandbox because
  Metal tests could not create a default device.
- Re-running `ctest --test-dir build --output-on-failure` outside the sandbox
  passed: 12/12 tests.

Current Module 2 gate status is summarized at the end of this file. This
checkpoint was enough for Assignments 2.1-2.4, but later assignments still had
to be verified before the module could pass.

## Assignment 2.5: Implement expand (Broadcast)

```cpp
View expand(const std::vector<ViewInt> &new_shape) const {
    ViewInt new_ndim = new_shape.size();
    ViewInt current_ndim = dim();
    
    ViewVector new_strides(new_ndim, 0);
    for (auto [i, j] = std::tuple(new_ndim - 1, current_ndim - 1); i >= 0; --i, --j) {
        size_t target_dim_size = new_shape[i];
        size_t current_dim_size = (j >= 0) ? shape[j] : 1;
    
        if (target_dim_size == current_dim_size) {
            new_strides[i] = (j >= 0) ? strides[j] : 1;
        } else if (current_dim_size == 1) {
            new_strides[i] = 0;
        } else {
            throw std::invalid_argument("expand shape is not valid for current shape");
        }
    }
    
    return {new_shape, new_strides, offset};
}
```

1. Because the expand dimension has stride `0` so when reading that dimension, we alaways read the same element (for example, `(3, 4)` with stride `(1, 0)` then physical_offset of `(1, i)` is always `1` for every `i` => reading the same element as in `(3, 1)`).
2. No, the Tensor isn't contiguous anymore because the new stride didn't match the ref_stride for new shape.

### Agent Feedback / Grading

Status: passed for Assignment 2.5's required behavior. `expand` now implements
stride-0 broadcasting for size-1 dimensions, and the focused test verifies the
row-repeat behavior through `physical_offset`.

What is good:

- The implementation aligns dimensions from the right, which is the right shape
  model for broadcasting and also supports leading broadcast dimensions such as
  `(3, 2) -> (3, 3, 2)`.
- Matching dimensions keep their old stride, and expanded size-1 dimensions get
  stride `0`, which is the core rule.
- `view.h` now includes `<tuple>` explicitly, so `std::tuple` is no longer
  relying on an indirect include.
- The test gathers physical offsets for `{3, 1}.expand({3, 4})` and checks the
  expected repeated pattern `{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2}`.
- The written answer correctly identifies why stride `0` reuses the same
  physical storage element, and it correctly says expanded views are generally
  not contiguous.

Remaining work:

- Invalid `expand` cases are now covered for non-1 dimension changes and
  attempted rank shrink.
- Consider rejecting negative or zero target dimensions explicitly. `ViewInt`
  allows negative shapes today, and `numel()` later casts shape entries through
  `size_t`, which is not a failure mode you want to discover downstream.
- The solution snippet should not show the `size_t` version of the reverse loop:
  `i >= 0` is wrong for unsigned integers. The actual implementation uses signed
  `ViewInt`, which is the safer version.
- Fix the small prose typo `alaways` in the written answer.

Verification:

- `cmake -S . -B build -DCOMTAM_BUILD_TESTS=ON` passed.
- `cmake --build build` passed.
- `ctest --test-dir build --output-on-failure -R "expand"` passed: 1/1 focused
  `expand` test.
- `ctest --test-dir build --output-on-failure -R "strides construction test|physical_offset test|permute and transpose|shrink|expand"` passed: 5/5 focused
  view tests.

Current Module 2 gate status is summarized at the end of this file. Assignment
2.5 is good enough; `reshape`, view-aware `Tensor::to_vector`, and the
movement-op gradient preview are graded below.

## Assignment 2.6: Implement reshape And Its Failure Message

>[!NOTE]
>`reshape` is allowed as a view only if it **preserves numel** and **the new view can produce the same physical_offset sequence without copying**.

```cpp
View reshape(const std::vector<ViewInt> &new_shape) const {
    ViewInt new_numel =
        std::reduce(new_shape.begin(), new_shape.end(), ViewInt{1}, std::multiplies<>());

    if (new_numel != numel()) {
        throw std::invalid_argument(
            "New shape cannot have different number of elements than current shape");
    }

    if (!is_contiguous()) {
        throw std::invalid_argument("Non-contiguous view cannot be reshaped without copy yet");
    }

    return {new_shape, offset};
}
```

Right now, `reshape` will throw on non-contiguous shape.
1. Important correction: `(2,3,4).permute((2,0,1)).reshape((4,6))` actually can be one view. The original contiguous strides are `(12,4,1)`. After `permute((2,0,1))`, the shape is `(4,2,3)` and the strides are `(1,12,4)`. The last two axes are still collapsible because `stride[1] == shape[2] * stride[2]`, or `12 == 3 * 4`. So `(4,2,3)` can become `(4,6)` with strides `(1,4)` and no copy.
2. A non-contiguous reshape is impossible only when the new logical order cannot be represented as `offset + sum(index[i] * stride[i])` with one stride per new axis. For example, a transposed `(2,3)` view has shape `(3,2)` and strides `(1,3)`. Its row-major logical physical-offset sequence is `0,3,1,4,2,5`. A 1-D reshape would need offsets `base + k * stride`, but this sequence has jumps `+3,-2,+3,-2,+3`, so no single 1-D stride can represent it.
3. For now, `comtam` can conservatively reject every non-contiguous reshape and ask the user to call a future `contiguous()` copy first. That is simpler than silently copying inside `reshape`, because silent copies hide cost.

### Agent Feedback / Grading

Status: partially passed for Assignment 2.6. The code supports contiguous
reshape as a pure view, preserves `numel`, and rejects non-contiguous reshape
instead of silently copying. However, it does not yet implement non-contiguous
reshapes that are still valid pure views.

What is good:

- `View::reshape` checks that the new shape preserves `numel`.
- Contiguous reshapes rebuild contiguous strides through the `View(new_shape,
  offset)` constructor.
- The implementation refuses non-contiguous reshape, which is an acceptable
  conservative first choice for comtam. Silent copies here would hide cost and
  confuse later autograd.
- Tests cover a valid contiguous reshape with an offset oracle, a truly
  impossible non-contiguous reshape failure, and a mismatched-element-count
  failure.

Remaining work:

- A representable non-contiguous reshape such as
  `(2,3,4).permute((2,0,1)).reshape((4,6))` is still not implemented. That is an
  acceptable conservative policy for now, but it is stricter than the full view
  semantics.
- The assignment asks for a failure message explaining which stride relationship
  broke. Your current message is clear enough for now, but it does not identify
  the broken stride relationship.

## Assignment 2.7: Make Readback View-Aware

```cpp
// we gather the data first
std::vector<scalar_t> result(numel);
for (size_t i = 0; i < numel; ++i) {
    result[i] = data[view_.physical_offset(i)];
}
device.copy<T>(result.data(), result.size(), *storage_.get());
```

### Agent Feedback / Grading

Status: passed for Assignment 2.7's required behavior. `Tensor::to_vector` now
gathers through `view_.physical_offset`, and the tensor-level transpose readback
test proves non-contiguous readback against a CPU oracle.

What is good:

- `Tensor::to_vector` now gathers from storage with
  `storage_->at<scalar_t>(view_.physical_offset(i))`, so the readback logic is
  view-aware in code.
- `Tensor::transpose` creates a metadata-only view sharing the same storage, and
  the readback test verifies `{0, 3, 1, 4, 2, 5}` for a transposed `(2, 3)`
  tensor.
- Tensor-level `shrink` and `expand` readback are now tested too, which verifies
  that `Tensor(storage, view)` can alias storage even when logical `numel`
  differs from the underlying storage size.
- `Tensor::from_vector` now rejects non-contiguous views with an exception rather
  than silently pretending to support scatter writes.

## Assignment 2.8: Movement Op Gradient Preview

| Forward op | Backward op |
| --- | --- |
| `reshape` | `reshape` back |
| `permute` | inverse `permute` |
| `expand` | `sum` over expanded (stride-0) dimensions |
| `slice` / `shrink` | `pad`|

Because each time the element `a[i]` is used for computation, it will have one gradient associated, so after that, we must accumulate total gradients it has.

### Agent Feedback / Grading

Status: passed for Assignment 2.8. The table has the right forward/backward
pairs, and the explanation captures the important point: an expanded value is
read multiple times, so backward must accumulate all contributions into the
single original element.

Tighten this later:

- Say the concrete count: if `a[i]` is read four times through a stride-0 axis,
  it receives the sum of four gradient contributions.
- Add the missing space before the final table pipe in `pad |`.

## Assignment 2.9: Read Magnetron / tinygrad Views

### Agent Feedback / Grading

Status: deferred / non-blocking for the Module 2 gate. I do not see the
required note titled "What comtam should copy from mature views, and what it
should postpone", so Assignment 2.9 itself is not passed. However, Module 2's
Exit Criteria do not list the mature-view reading note, so I am treating 2.9 as
a useful follow-up rather than a blocker for starting Module 3.

Still-useful deliverable:

- Answer what a mature view model stores that comtam does not.
- State whether it stacks views or keeps one view per tensor.
- Explain how padding is represented when stride math alone is not enough.
- Name one idea comtam should copy soon and one idea it should postpone.
- Record the first concrete case where one `View` per tensor becomes painful.

Verification:

- `cmake -S . -B build -DCOMTAM_BUILD_TESTS=ON` passed.
- `cmake --build build` passed.
- `ctest --test-dir build --output-on-failure -R "Tensor to_vector gathers through|Tensor from_vector rejects non-contiguous"` passed: 4/4 focused tensor-view tests.
- `ctest --test-dir build --output-on-failure -R "Tensor"` passed: 9/9 focused
  tensor tests.
- `ctest --test-dir build --output-on-failure -R "View"` passed: 6/6 focused
  view tests.
- `ctest --test-dir build --output-on-failure` passed: 18/18 tests.

Current verification:

- `cmake --build build` passed.
- `ctest --test-dir build --output-on-failure` passed: 18/18 tests.

Module 2 gate status: passed against the Exit Criteria. The important gate
requirements are satisfied:

- `View::physical_offset` is the single source of truth for logical-to-physical
  indexing.
- Contiguous, transposed, broadcasted, and sliced offset behavior is covered by
  focused tests.
- `Tensor::to_vector` respects strides and offsets, with tensor-level tests for
  transpose, shrink, and expand.
- `reshape` preserves `numel`, works for contiguous views, and fails loudly for
  unsupported non-contiguous views instead of silently copying.
- The `expand` backward explanation in Assignment 2.8 captures why gradient
  accumulation requires a reduction.

Sharp caveat: Assignment 2.6 is still conservative rather than complete for all
valid non-contiguous reshape views, and Assignment 2.9 remains deferred. Neither
blocks Module 3 under the written Exit Criteria, but both are worth revisiting
before the view system grows much more complicated.
