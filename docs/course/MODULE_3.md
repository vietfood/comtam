# Module 3: Eager Operator Dispatch

Do not turn every operator into its own little framework. Public tensor functions
should validate inputs, allocate outputs, then submit one uniform command.

## Target Shape

```text
add(x, y)
  -> validate dtype/device/shape
  -> allocate result
  -> Device::submit(Command{OpCode::Add, inputs, outputs})
  -> return result
```

This mirrors Magnetron's best idea: operation stubs are small, and execution is
centralized.

## Minimal Op Set

Implement:

- `fill`
- `add`
- `sub`
- `mul`
- `div`
- `neg`
- `relu`

## Assignment 3.1: Command Type

Create:

```cpp
enum class OpCode { Fill, Add, Sub, Mul, Div, Neg, Relu };

struct Command {
  OpCode op;
  std::span<Tensor*> inputs;
  std::span<Tensor*> outputs;
};
```

Use something close if spans do not fit your ownership layout.

## Assignment 3.2: Metal Kernel Library

Write simple Metal kernels for the minimal op set.

Keep indexing simple:

- assume contiguous tensors first
- one thread per output element
- no fusion
- no async complexity beyond command buffer commit/wait

## Assignment 3.3: One Dispatch Path

Implement:

```cpp
void Device::submit(const Command& cmd);
```

All elementwise ops must go through it.

## Completion Gate

You may start Module 4 only when:

- all minimal elementwise ops pass correctness tests
- every op uses `Device::submit`
- no op directly launches a kernel from its public tensor function
- all tests use host roundtrip oracles

