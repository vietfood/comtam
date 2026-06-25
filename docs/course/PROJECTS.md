# Final Projects

Choose one project after Module 7. Do not use final projects to justify adding
half the world to the framework.

## Project A: Linear Regression

Train:

```text
y = x @ w + b
```

Requirements:

- `Linear`
- `MSELoss`
- `SGD`
- loss decreases reliably

This is the smallest proof that tensors, kernels, autograd, and optimizer steps
are coherent.

## Project B: XOR MLP

Train a two-layer MLP on XOR.

Requirements:

- two `Linear` layers
- `relu` or `tanh`
- `MSELoss`
- `SGD`

This is the first proof that nonlinear composition works.

## Project C: MNIST MLP

Train a small MLP on MNIST.

Requirements:

- batched matmul
- simple data loader
- cross entropy or MSE one-hot baseline
- accuracy reporting

Do this only after XOR works. MNIST is not where the framework should learn
basic gradient hygiene.

## Project D: Better Matmul

Improve matmul with tiling.

Requirements:

- benchmark before and after
- preserve naive implementation as correctness reference
- test non-square shapes

This is a GPU project, not a framework architecture project.

## Project E: Private Buffers

Move from shared buffers to private GPU buffers where useful.

Requirements:

- explicit transfer path
- benchmark showing benefit
- no regression in host roundtrip tests

This should happen after correctness, not before.

