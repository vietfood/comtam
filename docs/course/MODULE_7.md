# Module 7: Modules, Parameters, Losses, And Optimizers

Now build the tiny training surface. Keep it smaller than you want.

## Minimal API

Implement:

- `Parameter`
- `Module`
- `Linear`
- `MSELoss`
- `SGD`

No module registry magic. No reflection. No serialization. A vector of parameter
pointers is enough.

## Assignment 7.1: Parameter

A `Parameter` is a tensor that requires grad by default.

It should not be a separate storage system.

## Assignment 7.2: Module

Implement a minimal base class:

```cpp
class Module {
public:
  virtual std::vector<Tensor*> parameters() = 0;
};
```

If a simpler approach works for the first examples, use it.

## Assignment 7.3: Linear

Implement:

```text
Linear(in_features, out_features)
forward(x) = x @ w + b
```

Use a simple random initializer.

## Assignment 7.4: SGD

Implement:

```text
p = p - lr * p.grad
```

Then implement `zero_grad`.

## Completion Gate

You may start Module 8 only when:

- linear regression loss decreases
- XOR loss decreases
- `zero_grad` is required and tested
- no optimizer state exists beyond learning rate

