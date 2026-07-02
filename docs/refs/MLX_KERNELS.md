# MLX Kernel Folder, Explained

This note exists because `refs/mlx/mlx/backend/metal/kernels/` and its sibling
`refs/mlx/mlx/backend/metal/` host files look like a maze on first read: 30+
headers, 20+ `.metal` files, a `jit/` folder, `jit_kernels.cpp` **and**
`nojit_kernels.cpp`, a `steel/` subtree, and a shell script that generates
C++ from headers. It is not accidental complexity - it is three genuinely
different mechanisms layered on top of each other, each solving a real
problem. This note untangles them one at a time.

Read [`MLX.md`](MLX.md) first for the overall architecture. This note is the
zoomed-in version of "Pass 3" and "Pass 6" from that reading order.

---

## 1. The Three Mechanisms, In One Table

| Mechanism | What it is | Where it lives | When it runs |
|---|---|---|---|
| **Template headers** | Generic Metal Shading Language kernel bodies, parameterized by `typename T`/`Op` | `kernels/*.h` | Never compiled by themselves; always `#include`d or embedded |
| **AOT instantiation catalogs** | Preprocessor macros that explicitly instantiate every needed (op, dtype, layout) combination | `kernels/*.metal` | Compiled once at **build time** by `xcrun metal`, linked into one `mlx.metallib` |
| **JIT instantiation strings** | The same explicit-instantiation text, but built as a C++ string at **run time** and compiled via `MTLDevice::newLibraryWithSource` | `jit/*.h` (format strings) + `jit_kernels.cpp` (assembly logic) | First use of a given kernel, only when `MLX_METAL_JIT=ON` |

The single most important fact: **`kernels/*.h` is the one and only source of
truth for kernel logic.** The `.metal` files and the JIT machinery are two
different *instancing* mechanisms over the same templates - never duplicated
logic. If you are hunting for "what does this kernel actually compute," go to
the `.h` file, not the `.metal` file.

---

## 2. Non-JIT (Default): Build-Time Instantiation

This is the default (`MLX_METAL_JIT` is `OFF` unless you pass
`-DMLX_METAL_JIT=ON` to CMake). It is also the simpler of the two paths to
understand first.

### 2.1 A `.metal` file is an instantiation catalog, not kernel logic

Take `kernels/unary.metal`. It does not define `unary_v`'s body - that lives
in `kernels/unary.h`:

```1:9:refs/mlx/mlx/backend/metal/kernels/unary.h
template <typename T, typename U, typename Op>
[[kernel]] void unary_v(
    device const T* in,
    device U* out,
    uint index [[thread_position_in_grid]]) {
  out[index] = Op()(in[index]);
}
```

`unary.metal` just declares, for every op and dtype pair, which concrete
instantiation of that template should exist as a real, callable Metal
function:

```8:11:refs/mlx/mlx/backend/metal/kernels/unary.metal
#define instantiate_unary_all(op, in_tname, out_tname, in_type, out_type)                 \
  instantiate_kernel("v_" #op #in_tname #out_tname, unary_v, in_type, out_type, op)       \
  instantiate_kernel("v2_" #op #in_tname #out_tname, unary_v2, in_type, out_type, op)     \
  instantiate_kernel("gn4_" #op #in_tname #out_tname, unary_g, in_type, out_type, op, 4)
```

then, further down, one line per op turns the crank for every dtype in that
op's supported set:

```34:34:refs/mlx/mlx/backend/metal/kernels/unary.metal
instantiate_unary_float(Exp)
```

`instantiate_unary_float` itself expands to `instantiate_unary_all` once per
floating dtype. So `instantiate_unary_float(Exp)` alone produces something
like `v_Expfloat16float16`, `v_Expfloat32float32`, `v_Expbfloat16bfloat16`,
each times three layout variants (`v_`, `v2_`, `gn4_`). One line, dozens of
real functions.

### 2.2 The `instantiate_kernel` macro is just explicit template instantiation

This is the one macro to understand cold, because everything else in the
kernels folder is built on top of it:

```18:24:refs/mlx/mlx/backend/metal/kernels/defines.h
// e.g. instantiate_kernel(binary_int, binary, a, b) ->
// [[host_name(binary_int)]] [kernel] binary<a, b>
#define instantiate_kernel(name, func, ...) \
  template [[host_name(                     \
      name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;
```

Nothing exotic: this is standard C++ explicit template instantiation
(`template void foo<int>(...)`) with two Metal-specific attributes bolted on:

- `[[kernel]]` marks the instantiation as a Metal compute entry point.
- `[[host_name("...")]]` gives it a *stable string name* independent of
  whatever the mangled template instantiation name would otherwise be. Host
  C++ code looks kernels up by this string, never by C++ symbol.

This is exactly the same trick comtam already uses by hand in
`comtam/kernels/binary.metal`:

```27:35:comtam/kernels/binary.metal
template [[ host_name("add_fp32") ]]
kernel void add(
    device const float*,
    device const float*,
    device float*,
    constant ViewInfo&,
    constant ViewInfo&,
    uint
);
```

MLX's `instantiate_kernel` is just a macro that writes that boilerplate for
you, parameterized over name/function/type-args, so one `.metal` file can
emit hundreds of these without hand-typing each one.

### 2.3 Build pipeline: `.metal` -> `.air` -> one `mlx.metallib`

`kernels/CMakeLists.txt` compiles each `.metal` file to an intermediate
`.air` object:

```8:14:refs/mlx/mlx/backend/metal/kernels/CMakeLists.txt
add_custom_command(
  COMMAND xcrun -sdk macosx metal ${METAL_FLAGS} -c ${SRCFILE}
          -I${PROJECT_SOURCE_DIR} -o ${TARGET}.air
  ...
)
```

then links every `.air` file into a single library:

```85:91:refs/mlx/mlx/backend/metal/kernels/CMakeLists.txt
add_custom_command(
  OUTPUT ${MLX_METAL_PATH}/mlx.metallib
  COMMAND xcrun -sdk macosx metallib ${KERNEL_AIR} -o
          ${MLX_METAL_PATH}/mlx.metallib
  DEPENDS ${KERNEL_AIR}
)
```

At runtime, `Device::Device()` loads that one file once
(`device.cpp:188-192`) and every kernel lookup for the rest of the process is
just a name-keyed search inside it plus a pipeline-state cache
(`Device::get_kernel`). `nojit_kernels.cpp` implements the `get_*_kernel`
API as a one-line pass-through to that lookup:

```15:22:refs/mlx/mlx/backend/metal/nojit_kernels.cpp
MTL::ComputePipelineState* get_unary_kernel(...) {
  return d.get_kernel(kernel_name);
}
```

This is structurally identical to comtam's `KernelLibrary::get(...)`
(`comtam/core/kernel.h`) - one `MTL::Library` loaded once, a
name-to-pipeline-state cache on top. MLX just has one library covering
thousands of pre-instantiated kernel names instead of a handful.

---

## 3. JIT (`MLX_METAL_JIT=ON`): Run-Time Instantiation

JIT mode exists purely to trade a smaller `mlx.metallib` for a small amount
of first-use compile latency. It reuses the exact same `kernels/*.h`
templates and the exact same instantiation idiom - it just performs the
instantiation with C++ string concatenation at runtime instead of the Metal
compiler at build time.

### 3.1 Step 1 (build time): embed a header's *source text* as a C++ string

`make_compiled_preamble.sh` runs the C preprocessor over a `kernels/*.h` file
(with `-DMLX_METAL_JIT` defined) and wraps the result in a raw string
literal:

```19:31:refs/mlx/mlx/backend/metal/make_compiled_preamble.sh
CONTENT=$($CC -I "$SRC_DIR" -DMLX_METAL_JIT -E -P "$INPUT_FILE" $CFLAGS 2>/dev/null)

cat << EOF > "$OUTPUT_FILE"
namespace mlx::core::metal {

const char* $SRC_NAME() {
  return R"preamble(
$CONTENT
)preamble";
}
```

CMake's `make_jit_source(unary)` wires this up for `kernels/unary.h`,
producing a generated `jit/unary.cpp` containing a function
`mlx::core::metal::unary()` that returns the entire header's text as a
string. `jit/includes.h` declares one such accessor per embeddable header:

```7:27:refs/mlx/mlx/backend/metal/jit/includes.h
const char* utils();
const char* unary_ops();
const char* unary();
const char* binary();
const char* reduce();
const char* softmax();
```

### 3.2 Step 2 (runtime): assemble source = preamble(s) + one explicit instantiation

`jit_kernels.cpp` implements `get_unary_kernel` by concatenating embedded
header text with a runtime-generated instantiation line, then handing the
whole string to the Metal compiler:

```39:59:refs/mlx/mlx/backend/metal/jit_kernels.cpp
MTL::ComputePipelineState* get_unary_kernel(...) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::unary_ops() << metal::unary();
    kernel_source << get_template_definition("v_" + lib_name, "unary_v", in_t, out_t, op);
    // ... v2_, gn4_ variants appended the same way ...
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}
```

`get_template_definition` (`kernels.h`) is the runtime, string-building
twin of the `instantiate_kernel` macro - it emits the exact same
`template [[host_name("...")]] [[kernel]] decltype(func<Args...>) func<Args...>;`
text, just via `fmt::format` instead of the preprocessor:

```198:218:refs/mlx/mlx/backend/metal/kernels.h
template <typename... Args>
std::string get_template_definition(std::string name, std::string func, Args... args) {
  return fmt::format(
      "\ntemplate [[host_name(\"{0}\")]] [[kernel]] decltype({1}) {1};\n",
      name, s.str());
}
```

For kernels whose instantiation needs more than one line of boilerplate
(softmax, steel GEMM), MLX keeps hand-written `fmt::format` template strings
in `jit/*.h` instead:

```3:9:refs/mlx/mlx/backend/metal/jit/arange.h
constexpr std::string_view arange_kernels = R"(
template [[host_name("{0}")]] [[kernel]] void arange<{1}>(
    constant const {1}& start,
    constant const {1}& step,
    device {1}* out,
    uint index [[thread_position_in_grid]]);
)";
```

### 3.3 Step 3 (runtime): compile once, cache twice

```350:373:refs/mlx/mlx/backend/metal/device.cpp
MTL::Library* Device::build_library_(const std::string& source_string) {
  auto options = MTL::CompileOptions::alloc()->init();
  options->setFastMathEnabled(false);
  options->setLanguageVersion(get_metal_version());
  auto mtl_lib = device_->newLibrary(ns_code, options, &error);
  return mtl_lib;
}
```

Two independent caches exist in `Device`:

- `library_map_`: keyed by a *library name* (e.g. `"Expfloat32float32"`),
  populated by `get_library(name, builder)` the first time any variant in
  that group is requested.
- `kernel_map_`: keyed by the *exact kernel name* (e.g.
  `"v_Expfloat32float32"`), populated by `get_kernel(name, lib)`.

Grouping matters: MLX deliberately compiles the `v_`, `v2_`, and `gn4_`
variants of one op+dtype together in a single `newLibraryWithSource` call
(`lib_name` strips only the layout prefix), so the *first* unary op you run
on `float32` pays one JIT compile for all three layout variants, and the
other two are free from then on.

### 3.4 Non-standard ops always JIT, regardless of the flag

`compiled.cpp` (fused elementwise graphs), `indexing.cpp` (gather/scatter),
and `hadamard.cpp` call `d.get_library(name, builder)` unconditionally,
because their kernel bodies depend on a runtime graph shape or a runtime size
that cannot be enumerated at build time. `MLX_METAL_JIT` only controls
whether the *finite, enumerable* op set (unary, binary, reduce, softmax,
sort, ...) is pre-baked into `mlx.metallib` or compiled on demand. This is
the same reason comtam's course explicitly avoids "compile a kernel per
runtime shape" as a v1 strategy - it is a real feature, but it is not free,
and MLX only reaches for it where enumeration is actually impossible.

---

## 4. Kernel Naming: The Protocol Between Host C++ and `[[host_name]]`

Every `get_*_kernel` function in MLX ends up building a plain `std::string`
that must exactly match a `[[host_name("...")]]` string somewhere in a
`.metal` file (AOT) or a JIT-generated instantiation (JIT). This string is
the entire interface between "C++ decided what to run" and "Metal has a
function with that name." A few representative patterns:

| Op family | Pattern | Example |
|---|---|---|
| Unary | `{layout}_{Op}{in_dtype}{out_dtype}` | `v_Expfloat32float32`, `gn4_Absfloat16float16` |
| Binary | `{layout}_{Op}{dtype}` | `vv_Addfloat32`, `g2_Multiplyfloat32` |
| Reduction | `{kind}_reduce_{variant}_{op}{dtype}` | `row_reduce_simple_sumfloat32` |
| Gather | `gather{dtype}{idx_dtype}_{n_idx}_{idx_ndim}` | `gatherfloat32int32_2_3` |
| Steel GEMM | `steel_gemm_fused_{ta}{tb}_{a_dtype}_{out_dtype}_bm{BM}_bn{BN}_bk{BK}_wm{WM}_wn{WN}` | `steel_gemm_fused_nn_float32_float32_bm64_bn64_bk16_wm2_wn2` |

`type_to_name(array)` (`backend/metal/utils.cpp`) produces the dtype tag
(`"float32"`, `"bfloat16"`, ...) used in nearly every name; a separate
function `get_type_string(Dtype)` produces the *actual MSL type* (`"float"`,
`"bfloat16_t"`) used inside template argument lists. These are easy to
confuse and worth keeping straight if you ever build something similar:
**one function names the kernel, a different function fills in its template
arguments.**

Steel GEMM goes one step further and separates the *pipeline cache key*
(`hash_name`, includes runtime boolean flags like `has_batch`) from the
*Metal function name* (`base_name`, stable across those flags), because the
flags are passed as Metal **function constants**
(`MTL::FunctionConstantValues`) rather than baked into the name - this avoids
recompiling a new function for every boolean combination while still caching
each specialized pipeline state separately.

---

## 5. Contiguous vs Strided: Why Every Op Family Has 3-5 Kernel Variants

The other reason the kernels folder looks big is that almost no op has just
one kernel - it has one *fast* kernel for the common contiguous case and one
or more *general* kernels for everything else. Using binary ops as the
clearest example:

```14:37:refs/mlx/mlx/backend/common/binary.h
enum class BinaryOpType {
  ScalarScalar,
  ScalarVector,
  VectorScalar,
  VectorVector,
  General,
};
```

- `ScalarScalar` / `ScalarVector` / `VectorScalar` / `VectorVector`: both
  inputs are either a single element or fully contiguous. Kernels
  (`binary_ss`, `binary_sv`, `binary_vs`, `binary_vv`) index with a flat
  `uint index [[thread_position_in_grid]]` - no shape/stride buffers needed
  at all.
- `General`: arbitrary strides/broadcasting. Before dispatch, host code calls
  `collapse_contiguous_dims(...)` to merge adjacent axes that are jointly
  contiguous (reducing effective `ndim`), then picks `g1`/`g2`/`g3`
  (specialized for 1-3 dims, shape encoded directly in a 3D dispatch grid) or
  `gn4` (arbitrary `ndim`, shape/strides passed as `constant` buffers, 4
  elements per thread).

This mirrors exactly what comtam's `ViewInfo` + `physical_offset(...)`
pattern is already doing by hand in `comtam/kernels/binary.metal` and
`comtam/kernels/utils.h` - the difference is that MLX additionally
special-cases the *common* layout (fully contiguous) into a much simpler,
faster kernel, instead of always paying the general stride-walk cost. That
split - "one dumb general-purpose kernel that always works" plus "one or two
fast-path kernels for the layouts that show up 99% of the time" - is the
single most reusable idea in this whole folder, independent of JIT/AOT or
naming conventions.

---

## 6. Folder Map Reference

```text
mlx/backend/metal/
|-- CMakeLists.txt          JIT/AOT source selection, make_jit_source()
|-- make_compiled_preamble.sh   .h -> embedded C++ string literal
|-- device.h / device.cpp   MTL::Device, library/pipeline load + cache
|-- kernels.h               get_*_kernel() declarations, get_template_definition
|-- jit_kernels.cpp         get_*_kernel() impls: JIT (MLX_METAL_JIT=ON)
|-- nojit_kernels.cpp       get_*_kernel() impls: AOT lookup (default)
|-- compiled.cpp            always-JIT fused elementwise graphs
|-- unary.cpp, binary.cpp, ternary.cpp   host dispatch: classify, name, encode
|-- reduce.cpp               host dispatch: ReductionPlan -> kernel family
|-- matmul.cpp                host dispatch: gemv vs steel routing
|-- indexing.cpp              always-JIT gather/scatter
|-- utils.h / utils.cpp       type_to_name, grid-size helpers
|-- jit/
|   |-- includes.h            declarations for embedded-source accessors
|   `-- arange.h, softmax.h, steel_gemm.h, ...   fmt-format instantiation strings
`-- kernels/
    |-- CMakeLists.txt         .metal -> .air -> mlx.metallib
    |-- defines.h              instantiate_kernel macro, tuning constants
    |-- utils.h                 elem_to_loc, grid/index helpers (device side)
    |-- unary.h / unary_ops.h / unary.metal        template / functors / catalog
    |-- binary.h / binary_ops.h / binary.metal
    |-- reduce.h / reduce.metal
    |-- reduction/              ops.h, reduce_all.h, reduce_row.h, reduce_col.h
    `-- steel/                  GEMM/conv template library (see MLX.md, Pass 5)
```

Reading rule of thumb: **`.h` = what it computes, host `.cpp` = which variant
and how it's launched, `.metal`/`jit/*.h` = how the variant gets a name.**
Once that split clicks, the folder stops looking like a maze and starts
looking like three short, boring layers repeated ~15 times for 15 op
families.
