# Module 6: Reverse-Mode Autograd

Autograd is a second graph system layered on top of eager execution.

Do not build a grand tape abstraction first. Store exactly what backward needs.

## Minimal Design

Each tensor may have:

```cpp
struct GradNode {
  OpCode op;
  std::vector<Tensor> inputs;
  std::vector<Attr> attrs;
  std::optional<Tensor> grad;
};
```

The exact ownership type is up to the implementation, but the rule is not:
outputs must keep their backward inputs alive.

## Assignment 6.1: Requires Grad

Add:

- `requires_grad`
- `grad`
- `zero_grad`
- `detach`

Only floating-point tensors may require gradients.

## Assignment 6.2: Record Forward Ops

When an op has a backward rule and any input requires grad:

1. mark the output as requiring grad
2. store op code
3. store needed inputs
4. store needed attrs

## Assignment 6.3: Backward Traversal

Implement scalar `backward()`:

1. require scalar root
2. topologically sort dynamic graph
3. seed root grad with `1`
4. traverse reverse topological order
5. call local backward rules
6. accumulate gradients

## Assignment 6.4: Local Gradient Rules

Implement backward for:

- `add`
- `sub`
- `mul`
- `div`
- `neg`
- `relu`
- `sum`
- `mean`
- `matmul`
- `reshape`
- `transpose`
- `broadcast_to`

## Completion Gate

You may start Module 7 only when:

- finite-difference or oracle gradient checks pass for every listed op
- gradient accumulation works when a tensor is used twice
- broadcasting backward reduces to the original input shape
- in-place ops are still out of scope

