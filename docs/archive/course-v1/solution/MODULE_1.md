# Assignment 1.1: Trace Storage Lifetime

1. `Storage` should not know `DType` or manage the `DType` of `Tensor`. Keep it
   as a dumb byte buffer.
2. `sizeof(T)` in `Device::copy<T>` must be guaranteed to match the tensor
   dtype. Use dtype dispatch, similar to Legrad or ATen, to recover the correct
   C++ scalar type from `DType`.

```cpp
// clang-format off
#define COMTAM_DISPATCH_DTYPE(DTYPE, ...) \
    [&] { \
        switch (DTYPE) { \
        case comtam::core::DType::Float32: { \
            using scalar_t = float; \
            return __VA_ARGS__(); \
        } \
        default: \
            throw std::runtime_error("unsupported dtype"); \
        } \
    }()
```

3. If a second dtype is added later, every tensor host-transfer path must stay
   routed through dtype dispatch. Otherwise a caller could accidentally copy
   bytes using the wrong C++ scalar type.

## Agent Feedback / Grading

Status: passed.

Good call keeping `Storage` dtype-agnostic. That keeps ownership separate from
interpretation: `Storage` owns bytes, while `Tensor` owns dtype and view
metadata. The `COMTAM_DISPATCH_DTYPE` macro is acceptable at this stage because
there is only one dtype and it keeps the future extension point visible.

Evidence:

- `Tensor` constructors, `from_vector`, and `to_vector` dispatch from `DType` to
  `scalar_t`.
- `Device::copy` rejects byte-count mismatches before `memcpy`.
- Tests cover float32 host round-trip and storage byte-count mismatch rejection.

Caveat: the macro currently only supports `Float32`, which is correct for the
course boundary. Do not add more dtype machinery until a real second dtype
forces it.

## Assignment 1.2: Prove Sharing Work

Create a new API:
```cpp
Tensor(const std::shared_ptr<core::Storage> &storage, const std::vector<core::ViewInt> &shape,
    core::DType dtype = core::DType::Float32)
    : dtype_(dtype), view_(shape), storage_(nullptr) {
    COMTAM_DISPATCH_DTYPE(dtype_, [&] {
        // we must ensure the storage byte match the Tensor
        if (sizeof(scalar_t) * view_.numel() != storage->size()) {
            throw std::runtime_error("[ERROR] Storage size doesn't match dtype and shape");
        }
        // then we only need to set target
        storage_ = storage;
    });
}
```

Then this test must be passed:
```cpp
TEST_CASE("Two tensor headers can share one Storage safely", "[tensor][metal]") {
    core::Context context;
    auto &device = context.device();

    auto input = utils::generate_random_array<float>(kSize, 0.0F, 1.0F);
    auto shape = std::vector<core::ViewInt>{static_cast<core::ViewInt>(kSize)};

    std::unique_ptr<Tensor> survivor = nullptr;

    {
        auto storage = std::make_shared<core::Storage>(device.allocate(kSize * sizeof(float)));
        device.copy(input.data(), input.size(), *storage);

        Tensor a(storage, shape, core::DType::Float32);
        survivor = std::make_unique<Tensor>(storage, shape, core::DType::Float32);

        require_all_close(input, a.to_vector<float>(device));
        require_all_close(input, survivor->to_vector<float>(device));
    }

    require_all_close(input, survivor->to_vector<float>(device));
}
```

## Agent Feedback / Grading

Status: passed.

The public constructor from `std::shared_ptr<core::Storage>` proves the intended
header/storage split directly. The byte-size check is the right guard: two
headers may interpret the same storage only when dtype and shape imply exactly
the storage byte count.

Evidence:

- `Tensor` stores `std::shared_ptr<core::Storage>`, so the buffer stays alive
  until the last tensor header releases it.
- The survivor test destroys one tensor header and still reads the original
  data through the other.
- The shared-write test writes through one tensor and reads the replacement data
  through another tensor over the same storage.

Caveat: in-flight GPU lifetime is currently simple because `Device::submit`
waits synchronously with `waitUntilCompleted()`. Revisit this when async command
submission exists.

## Assignment 1.3: Keep Storage::print Small

```cpp
template <typename T> void print() const {
    const std::string label = "Storage with bytes=" + std::to_string(size_) + "\nContent";
    comtam::utils::print_array(static_cast<T *>(buffer_->contents()), size_ / sizeof(T), label);
}
```

## Agent Feedback / Grading

Status: passed.

`Storage::print<T>` stayed small and obviously debug-only. It does not mutate
the buffer, allocate framework state, or become a logging abstraction.

Caveat: printed arrays are debugging aids, not correctness evidence. The pass
comes from the tests above, not from `print`.

## Module 1 Verdict

Status: passed.

Verification:

```text
compiled        -> passed with `cmake --build build`
smoke ran       -> passed outside sandbox with `ctest --test-dir build --output-on-failure`
gate passed     -> passed
```

Commands run:

```text
cmake -S . -B build -DCOMTAM_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Note: CTest failed inside the sandbox because Metal could not create the default
device. The same CTest command passed outside the sandbox with 8/8 tests green.
