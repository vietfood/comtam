## Module 0: Understanding The Current Codebase

Before implementing anything new, you need to deeply understand what already
exists and what it cannot yet do.

## Why This Module Exists

This module prevents fake progress.

comtam already has a `Context`, a `Device`, a `Storage`, a `View`, a
`KernelLibrary`, a `Command`, and a `Tensor` that can copy data to and from the
GPU. Some of it is incomplete, and `main.cpp` currently does not run a kernel at
all. That is fine. Your job in this module is not to fix it. Your job is to
answer one question precisely:

```text
What happens when a tensor moves from host memory to a Metal buffer and back?
```

If you cannot answer that with file and line references, every later module will
be guessing.

Small fixes are allowed only when they unblock tracing. Everything else goes
into [`BUG_LEDGER.md`](BUG_LEDGER.md) in Assignment 0.5.

## The Current Shape

Read these files in this order and keep a notebook open:

```text
comtam/core/context.h        Context owns Device and KernelLibrary
comtam/core/device.h/.cpp    Device owns MTL::Device and MTL::CommandQueue
comtam/core/storage.h/.cpp   Storage owns one MTL::Buffer
comtam/core/view.h           View holds shape, strides, offset
comtam/core/dtype.h          DType, currently float32 only
comtam/core/ops.h            Op enum and Command struct
comtam/core/kernel.h/.cpp    KernelLibrary compiles and caches pipelines
comtam/tensor.h/.cpp         Tensor ties View + Storage + DType together
comtam/main.cpp              the current entry point
```

The intended runtime ownership is:

```text
Context
  owns Device          (MTL::Device, MTL::CommandQueue)
  owns KernelLibrary   (MTL::Library, pipeline cache)

Tensor
  has DType
  has View             (shape, strides, offset)
  shares Storage       (std::shared_ptr<Storage> -> MTL::Buffer)
```

This is the Magnetron arc narrowed to one backend. See
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) for the intended layering and
[`../NOTE.md`](../NOTE.md) for metal-cpp ownership rules.

### Assignment 0.1: Trace Tensor Construction From Host Data ⭐

**Task:** Trace what happens when you run, against the current `main.cpp`:

```cpp
core::Context context;
auto& device = context.device();

std::vector<float> A = comtam::utils::generate_random_array<float>(N, 0, 1);
Tensor tA(A.data(), {N}, device);
```

**Questions to answer:**

1. What does `View({N})` compute for `shape`, `strides`, and `offset`?
2. How many bytes does `device.allocate(...)` request, and where does the byte
   count come from?
3. What does `device.copy(data, count, buffer)` actually do at the Metal level?
4. Who owns the `MTL::Buffer` after the constructor returns, and through how many
   layers of indirection?

**Hint:** Follow `Tensor::Tensor(const float*, ...)` in `comtam/tensor.cpp:7`
into `Device::allocate`, the `Device::copy` template in `comtam/core/device.h:36`,
and `Storage` in `comtam/core/storage.h`.

**Deliverable:** A short trace document with file and line references for every
step from host vector to Metal buffer.

**Why this assignment:** Construction is the smallest path that still touches
every ownership layer: Context, Device, Storage, View, and the metal-cpp shared
buffer. If you understand this, allocation never feels like magic again.

---

### Assignment 0.2: Trace The Round Trip ⭐

**Task:** Trace the readback path:

```cpp
auto C = tA.to_vector(device);
```

**Questions to answer:**

1. How does `to_vector` know how many elements to read?
2. `Device::copy(Storage&, T*, count)` uses `std::memcpy` from
   `buffer->contents()`. Why is that safe here, and what assumption about the
   buffer's storage mode does it depend on?
3. What would break if the tensor were not contiguous? (You will not fix this
   yet, just predict it.)

**Hint:** Read `Tensor::to_vector` in `comtam/tensor.cpp:43` and the second
`Device::copy` overload in `comtam/core/device.h:45`.

**Insight to discover:** `to_vector` currently trusts `numel()` and a shared
storage mode. It ignores strides and offset entirely. Write down exactly where
that assumption is hidden.

**Why this assignment:** A round trip with no kernel is the cheapest possible
correctness check. If host -> buffer -> host does not preserve data, nothing
above it can be trusted.

---

### Assignment 0.3: Run The First Kernel Path ⭐⭐

**Task:** The current `main.cpp` never submits a command. Trace the dispatch
machinery that already exists but is unused:

```cpp
struct Command {            // comtam/core/ops.h
    Op op;
    Storage* a;
    Storage* b;
    Storage* out;
    size_t elements;
};
```

**Questions to answer:**

1. `Device::submit(const Command&, KernelLibrary&)` is declared in
   `comtam/core/device.h:55`. Read its definition in `device.cpp`. What is the
   full sequence: pipeline lookup, command buffer, encoder, buffer binding,
   threadgroup sizing, commit, wait?
2. How does `KernelLibrary::get(const Op&)` map an `Op` to a compiled
   `MTL::ComputePipelineState`? Where does the kernel name come from?
   (See `op_to_kernel_name` in `comtam/core/ops.h:17`.)
3. What does `comtam/kernels/binary.metal` declare, and do the kernel function
   names match `op_to_kernel_name`?
4. How are threads-per-grid and threads-per-threadgroup chosen, and what happens
   when `elements` is not a multiple of the threadgroup size?

**Deliverable:** A diagram of the submit path from `Command` to GPU execution,
plus a note on whether `binary.metal` and `op_to_kernel_name` agree.

**Why this assignment:** Elementwise add is the smallest example that still
passes through the whole runtime: op selection, pipeline lookup, buffer binding,
launch, and synchronization. This is the spine of Module 3.

---

### Assignment 0.4: Build And Smoke Run ⭐

**Task:** Build comtam with CMake and run the current executable.

**Steps:**

1. Configure and build with CMake (see `CMakeLists.txt` and
   `comtam/CMakeLists.txt`).
2. Confirm where `.metal` sources are copied at build time and whether a
   `default.metallib` is produced.
3. Run the executable. Record whether it compiled, whether it ran, and what it
   printed.

**Reporting rule:** Separate the three claims:

```text
compiled   -> build succeeded
smoke ran  -> executable ran without crashing
gate met   -> not yet; Module 0 has no correctness gate beyond the round trip
```

If Metal execution fails inside a sandbox, say so explicitly and only run
outside the sandbox with permission. See [`../NOTE.md`](../NOTE.md) for command
buffer error checking.

**Why this assignment:** You cannot grade later modules honestly if you cannot
reliably build and run the current one.

---

### Assignment 0.5: Start The Bug Ledger ⭐⭐

**Task:** Run comtam until something is wrong, missing, or surprising, then turn
each finding into a small classified entry in [`BUG_LEDGER.md`](BUG_LEDGER.md).

The goal is not to fix everything in Module 0. The goal is to rebuild your
mental model by forcing the code to show where it is fragile or unfinished.

Probe at least these:

```text
# round trip with no kernel
Tensor t(A.data(), {N}, device); auto out = t.to_vector(device);  // oracle: A

# the unused dispatch path
build a Command for Op::ADD and call device.submit(...)           // does it run?

# name agreement
do binary.metal kernel names match op_to_kernel_name?             // oracle: read both

# non-contiguous readback (predicted failure)
imagine a transposed view, then to_vector                         // oracle: reasoning
```

For each finding, record an oracle and a first guess at the owning concept. Use
the entry template in [`BUG_LEDGER.md`](BUG_LEDGER.md).

Rules:

- Reduce every finding to the smallest repro you can.
- Do not fix more than one thing per patch.
- Do not mix a fix with cleanup.
- If something is an unimplemented feature rather than a bug, say so. Do not
  pretend it is a bug.

**Why this assignment:** comtam is young, so most findings here will be "not
implemented yet" rather than "wrong". Classifying them now turns vague unease
into an owned, ordered queue for Modules 1 through 9.

## Module 0 Checklist

- [ ] 0.1 Trace tensor construction from host data.
- [ ] 0.2 Trace the host -> GPU -> host round trip.
- [ ] 0.3 Trace the unused `Command` dispatch path.
- [ ] 0.4 Build with CMake and smoke run the executable.
- [ ] 0.5 Start the bug ledger with classified entries.

## Exit Criteria

You are ready for Module 1 when:

1. You can name the owner of every Metal object (device, queue, buffer, library,
   pipeline) with a file and line reference.
2. You can describe the full host -> buffer -> host round trip without reading
   the code again.
3. You can describe the `Command` -> `Device::submit` -> GPU path, even though
   `main.cpp` does not call it yet.
4. The bug ledger has at least a few classified entries separating "not
   implemented" from "wrong".
