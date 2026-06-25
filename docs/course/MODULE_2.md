# Module 2: Tensor Metadata, Shapes, Strides, And Views

A tensor is not just memory. It is memory plus an interpretation.

## Mental Model

```text
physical_offset = storage_offset + sum(index[d] * stride[d])
```

That formula must become boring before you write more kernels.

## Minimal Tensor

Implement:

```cpp
enum class DType { Float32 };

struct Shape {
  std::vector<int64_t> dims;
};

class Tensor {
  DType dtype;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  int64_t storage_offset;
  std::shared_ptr<Storage> storage;
};
```

If you dislike `std::shared_ptr`, make a deliberate alternative. But views need
shared storage ownership somehow. Pretending they do not is how memory bugs
enter.

## Assignment 2.1: Contiguous Tensors

Implement:

- `numel`
- contiguous stride construction
- `is_contiguous`
- host `from_vector`
- host `to_vector`

## Assignment 2.2: Views

Implement:

- `as_strided`
- `reshape` for contiguous tensors
- `transpose`
- 1D and 2D slice

Do not implement a generic Magnetron-style stride solver yet. First earn it with
tests.

## Assignment 2.3: Offset Tests

For each view, test logical-to-physical offset mapping:

- contiguous `(2, 3)`
- transpose `(2, 3) -> (3, 2)`
- slice rows
- slice columns
- reshape contiguous `(2, 3) -> (6)`

## Completion Gate

You may start Module 3 only when:

- views do not copy storage
- offset tests pass
- reshape rejects non-contiguous tensors
- every view has a clear storage owner

