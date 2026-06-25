# Module 4: Broadcasting And Reductions

Broadcasting and reductions are the first place tensor semantics stop being
obvious.

## Learning Goals

By the end of this module, you should be able to explain:

1. Why broadcasting aligns dimensions from the right.
2. Why broadcast views use stride `0`.
3. Why reductions are not elementwise kernels.
4. Why backward broadcasting later needs reduce-back.

## Assignment 4.1: Broadcast Shapes

Implement:

```cpp
std::vector<int64_t> broadcast_shape(a.shape, b.shape);
```

Reject incompatible shapes with a useful error.

## Assignment 4.2: Broadcast Views

Implement `broadcast_to`.

Rules:

- introduced dimensions get stride `0`
- dimensions expanding from `1` get stride `0`
- non-expanded dimensions keep their original stride

## Assignment 4.3: Elementwise Broadcast Kernels

Update elementwise kernels to support non-contiguous and broadcasted inputs.

Do this with explicit shape/stride metadata. Do not create hidden contiguous
copies.

## Assignment 4.4: Reductions

Implement:

- `sum`
- `mean`

Start with reduce-all. Add single-axis reductions only after reduce-all is
solid.

## Completion Gate

You may start Module 5 only when:

- broadcasted `add`, `mul`, and `div` pass tests
- reduce-all `sum` and `mean` pass tests
- stride-0 behavior is tested directly
- reduction kernels are separate from elementwise kernels

