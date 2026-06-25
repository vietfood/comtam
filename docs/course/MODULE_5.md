# Module 5: Matmul And Layout

Matmul is not just another binary op. It has layout, reduction, and performance
pressure all at once. Start dull and correct.

## Scope

Implement 2D `float32` matmul:

```text
(M, K) @ (K, N) -> (M, N)
```

Do not implement batched matmul, transposed fast paths, tiling wizardry, or
half precision yet.

## Assignment 5.1: Shape Validation

Reject invalid matmuls before kernel launch.

Error messages should include both shapes and the mismatched dimensions.

## Assignment 5.2: Naive Metal Matmul

Write a basic kernel:

- one output element per thread
- loop over `K`
- support contiguous inputs first

## Assignment 5.3: Linear Forward

Implement the forward computation:

```text
y = x @ w + b
```

This is not `nn.Linear` yet. It is a direct tensor-level proof.

## Assignment 5.4: Layout Notes

Write a short note explaining:

- what row-major layout means
- how `transpose` changes strides
- why this naive matmul rejects non-contiguous inputs for now
- what would be needed to support transposed weights without copying

## Completion Gate

You may start Module 6 only when:

- 2D matmul passes correctness tests
- `x @ w + b` matches an oracle
- non-contiguous inputs are either correctly supported or explicitly rejected
- the layout note exists

