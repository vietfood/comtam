# Magnetron Study Guide

This note is a map for studying the `refs/magnetron/` subtree inside this repository.
It complements the MLX notes ([`MLX.md`](MLX.md), [`MLX_KERNELS.md`](MLX_KERNELS.md),
[`MLX_REDUCTION.md`](MLX_REDUCTION.md)): Magnetron is the small, readable,
eager, C-native runtime reference. MLX is the "how does a serious, lazy,
compiled Apple framework get performance" reference. Read Magnetron first for
runtime shape - it maps almost 1:1 onto what `comtam` is building. Read MLX
second for kernel-organization scale.

---

## What Magnetron Is

Magnetron is built as a small ML runtime with four clear layers:

1. Python user API and training utilities.
2. Nanobind bindings that expose the native runtime.
3. A C core that owns tensors, views, operators, autodiff, snapshots, and backend dispatch.
4. Backend shared libraries such as CPU and CUDA.

---

## Architecture In One Diagram

```text
Python examples / nn / optim
        |
        v
python/magnetron/__init__.py
        |
        v
magnetron/bindings/_module.cpp + tensor bindings
        |
        v
include/magnetron/magnetron.h
        |
        v
core runtime
  - mag_context
  - mag_tensor
  - mag_op_stubs
  - mag_autodiff
  - mag_snapshot
  - mag_backend registry
        |
        v
active device submit(...)
        |
        +--> magnetron_cpu
        |
        +--> magnetron_cuda
```

The important point is that Python is thin, the core is explicit, and the backend boundary is a real ABI boundary.

---

## The Core Ideas

### 1. `mag_context_t` owns the runtime

Start with `mag_context_t` in `magnetron/magnetron/core/mag_context.h`. The
exact struct is small and worth reading in full - nothing about the runtime
is hidden behind an opaque handle:

```38:52:refs/magnetron/magnetron/core/mag_context.h
struct mag_context_t {
  uintptr_t tr_id;
  mag_context_flags_t flags;
  mag_dtype_t default_dtype;
  mag_machine_info_t machine;
  mag_rt_telemetry_t telemetry;
  mag_slab_alloc_t tensor_slab;
  mag_slab_alloc_t storage_slab;
  mag_slab_alloc_t view_meta_slab;
  mag_slab_alloc_t au_state_slab;
  mag_backend_registry_t *backend_registry;
#ifdef MAG_DEBUG
  mag_tensor_t *alive_head;
#endif
};
```

The context owns:

- a thread affinity id (`tr_id`) - tensors must be created on the thread that
  created the context; there is no cross-thread tensor sharing to reason
  about,
- machine information and telemetry counters,
- **four separate slab allocators** for tensor headers, storages, view
  metadata, and autodiff state - one obvious owner per object kind, no
  general-purpose heap churn on the hot path,
- the backend registry and, through it, every loaded device.

`mag_ctx_create()` makes the model explicit end to end:

```131:146:refs/magnetron/magnetron/core/mag_context.c
mag_context_t *ctx = (*mag_alloc)(NULL, sizeof(*ctx), 0);
memset(ctx, 0, sizeof(*ctx));
mag_slab_init(&ctx->tensor_slab, sizeof(mag_tensor_t), __alignof(mag_tensor_t), 0x1000);
mag_slab_init(&ctx->storage_slab, sizeof(mag_storage_buffer_t), __alignof(mag_storage_buffer_t), 0x1000);
mag_slab_init(&ctx->view_meta_slab, sizeof(mag_view_meta_t), __alignof(mag_view_meta_t), 0x1000);
mag_slab_init(&ctx->au_state_slab, sizeof(mag_au_state_t), __alignof(mag_au_state_t), 0x1000);
ctx->tr_id = mag_thread_id();
ctx->default_dtype = MAG_DTYPE_FLOAT32;
ctx->flags |= MAG_CTX_FLAG_GRAD_RECORDER;
mag_machine_info_probe(&ctx->machine);
ctx->backend_registry = mag_backend_registry_init(ctx);
```

Notable details:

- gradient recording is **on by default** - the cost of tracking a tape is
  accepted up front, the same posture comtam should probably take rather
  than making autograd an opt-in mode to special-case around,
- backends are discovered dynamically at startup by scanning for
  `magnetron_<backend>.{so|dylib|dll}` shared libraries next to the core
  library (`cpu` is required, `cuda`/`custom` are optional),
- destroy time does leak detection: an `alive_head` linked list in debug
  builds is walked and will panic on outstanding tensors unless explicitly
  suppressed.

This is not a global hidden runtime. It is a concrete object that owns
almost everything, and every subsystem below hangs off one of its four
slabs or its backend registry.

### 2. `mag_tensor_t` is a header plus storage

Study `magnetron/magnetron/core/mag_tensor.h` and
`magnetron/magnetron/core/mag_tensor.c` next. The exact field list:

```52:68:refs/magnetron/magnetron/core/mag_tensor.h
struct mag_tensor_t {
  MAG_RC_INJECT_HEADER;           /* mag_rc_control_block_t __rcb */
  mag_context_t  *ctx;
  mag_coords_t coords;            /* rank + shape[16] + strides[16] */
  mag_dtype_t dtype : 8;
  mag_tensor_flags_t flags : 8;   /* IS_VIEW | IS_GRAD | REQUIRES_GRAD */
  mag_storage_buffer_t *storage;
  int64_t numel;
  int64_t storage_offset;         /* element offset into storage */
  mag_view_meta_t *view_meta;
  mag_au_state_t *au_state;
  uint64_t version;
#ifdef MAG_DEBUG
  mag_tensor_t *alive_next;
#endif
};
```

This is a very runtime-oriented tensor design. A tensor is not a symbolic
node first. It is a concrete runtime object with storage, view state, and
optional gradient metadata.

Two details matter:

- Views are implemented with `mag_as_strided(...)` and share storage.
- The tensor header is reference counted separately from storage - via
  `MAG_RC_INJECT_HEADER`, which expands to an atomic refcount plus a
  destructor function pointer embedded directly in the struct:

```22:28:refs/magnetron/magnetron/core/mag_rc.h
typedef struct mag_rc_control_block_t {
  #ifdef MAG_DEBUG
    uint32_t __sentinel;
  #endif
  volatile mag_atomic32_t rc_strong;
  mag_status_t (*dtor)(void *);
} mag_rc_control_block_t;
```

Storage (`mag_storage_buffer_t`) is its own independently reference-counted
object, defined in `mag_backend.h` rather than `mag_tensor.h` - it belongs
to the device/backend boundary, not to the tensor's own module:

```42:54:refs/magnetron/magnetron/core/mag_backend.h
struct mag_storage_buffer_t {
  MAG_RC_INJECT_HEADER;
  mag_context_t *ctx;
  mag_storage_flags_t flags;
  uint32_t alignment;
  uintptr_t base;
  size_t size;
  mag_device_t *device;
  union { void *impl; uint8_t intrusive_storage[...]; } aux;
};
```

That separation - one refcount on the header, one refcount on the storage,
storage owned conceptually by whichever device allocated it - is the major
part of how Magnetron stays explicit while still supporting slicing,
reshaping, and aliasing without use-after-free bugs. A tensor's data pointer
is always `storage->base + storage_offset * dtype_size`; there is no other
path to the bytes.

### 3. Views are first-class and solved in the runtime

The view path lives mostly in:

- `magnetron/magnetron/core/mag_tensor.c`
- `magnetron/magnetron/core/mag_op_stubs.c`
- `magnetron/magnetron/core/mag_coords.c` / `mag_coords.h`

`mag_as_strided(...)` is the one function that actually creates a view. It
allocates a **new header**, shares the base tensor's storage via an incref
(never a copy), and either creates or shares `view_meta`:

```151:167:refs/magnetron/magnetron/core/mag_tensor.c
mag_tensor_t *tensor = mag_tensor_init_header(ctx, base->dtype, rank, numel);
// copy shape/strides
tensor->storage = base->storage;
mag_rc_incref(base->storage);
tensor->storage_offset = offset;
if (!(base->flags & MAG_TFLAG_IS_VIEW))
  tensor->view_meta = mag_view_meta_alloc(base);
else {
  tensor->view_meta = base->view_meta;
  mag_rc_incref(tensor->view_meta);
}
tensor->flags = base->flags | MAG_TFLAG_IS_VIEW;
```

`mag_view_meta_t` itself is tiny - just a refcounted pointer back to the
base tensor plus a version snapshot, used to detect "the base tensor was
mutated in place after this view was taken" bugs.

The rest of the design is practical:

- if a view is obvious, reuse strides directly,
- if the tensor is contiguous (`mag_coords_contiguous`, below), synthesize
  row-major strides,
- otherwise, call the generic stride solver (`mag_solve_view_strides`),
- if a reshape cannot be a view, materialize with `contiguous()` and rebuild
  shape.

Shape/stride metadata itself is a flat, fixed-size struct - no heap
allocation per tensor for shape data, capped at `MAG_MAX_DIMS = 16`:

```21:25:refs/magnetron/magnetron/core/mag_coords.h
typedef struct mag_coords_t {
  int64_t rank;
  int64_t shape[MAG_MAX_DIMS];
  int64_t strides[MAG_MAX_DIMS];
} mag_coords_t;
```

and contiguity is a direct, size-1-dimension-aware reverse walk, worth
comparing line for line against comtam's own contiguity check:

```92:101:refs/magnetron/magnetron/core/mag_coords.c
bool mag_coords_contiguous(const mag_coords_t *x) {
  int64_t y=1;
  for (int64_t i=x->rank-1; i >= 0; --i) {
    if (x->shape[i] == 1) continue;
    if (x->strides[i] != y) return false;
    y *= x->shape[i];
  }
  return true;
}
```

A second, kernel-facing struct - `mag_coords_iter_t` in
`mag_coords_iter.h` - exists purely so hot loops don't dereference through a
`mag_tensor_t*`/`mag_coords_t` chain per element; it's a flat copy of
rank/shape/strides plus inline helpers (`mag_coords_iter_to_offset`,
`mag_coords_iter_broadcast`) for turning a linear index into a strided
offset. Worth noting: this header is also tagged `MAG_CUDA_DEVICE`, meaning
the exact same index-math source compiles for both CPU kernels and CUDA
device code - one indexing implementation, not two.

### 4. Operators are eager, validated in core, and dispatched to the active device

The heart of execution is `magnetron/magnetron/core/mag_op_stubs.c`, and the
single most important function in the whole codebase to read closely is
`mag_dispatch`. Every op - unary, binary, view, reduction, matmul - funnels
through it:

```183:206:refs/magnetron/magnetron/core/mag_op_stubs.c
static mag_status_t mag_dispatch(...) {
  if (!!(ctx->flags & MAG_CTX_FLAG_GRAD_RECORDER) && meta->backward) {
    for (uint32_t i=0; i < num_out; ++i) {
      mag_au_state_t *au = mag_au_state_lazy_alloc(&r->au_state, r->ctx);
      au->op = op;
      for (uint32_t j=0; j < num_in; ++j) {
        au->op_inputs[j] = in[j];
        if (input->flags & MAG_TFLAG_REQUIRES_GRAD && !(r->flags & MAG_TFLAG_REQUIRES_GRAD))
          mag_try(mag_tensor_set_requires_grad(err, r, true));
        mag_rc_incref(input);
      }
      if (params) memcpy(au->op_attrs, params, ...);
    }
  }
  mag_command_t cmd = { .op=op, .in=in, .out=out, .num_in=num_in, .num_out=num_out };
  mag_status_t stat = (*device->submit)(device, err, &cmd);
  ++ctx->telemetry.ops_dispatched;
  return stat;
}
```

The typical calling pattern, seen in every op stub above `mag_dispatch`, is:

1. Validate dtype and shape semantics in core.
2. Allocate the result tensor (`mag_empty`/`mag_empty_like`), or create a
   view (`mag_as_strided`) for shape-only ops.
3. Call `mag_dispatch`, which records autodiff metadata **on the output
   tensor** if gradient recording is enabled, then submits a
   `mag_command_t` to `device->submit`.

Three representative call sites make the pattern concrete. A unary op
(`mag_neg`, `mag_relu`, generated via `mag_impl_unary_pair`):

```965:977:refs/magnetron/magnetron/core/mag_op_stubs.c
static mag_status_t mag_op_stub_unary(...) {
  mag_try(mag_check_dtype_and_device_compat(err, op, &x, 0));  /* validate */
  if (inplace) {
    mag_try(mag_tensor_strided_view(err, &result, x));
    mag_try(mag_check_inplace_grad_ok(err, x));
  } else {
    mag_try(mag_empty_like(err, &result, x));                  /* allocate */
  }
  mag_try_or(mag_dispatch(err, op, inplace, layout, &x, 1, &result, 1), ...);
  *out_result = result;
  return MAG_STATUS_OK;
}
```

A binary op (`mag_add`) additionally has to broadcast shapes and promote
dtypes before it ever reaches `mag_dispatch`:

```1252:1266:refs/magnetron/magnetron/core/mag_op_stubs.c
mag_coords_broadcast_shape(&x->coords, &y->coords, dims, &rank);
mag_try(mag_empty(err, &result, x->ctx, res_type, rank, dims, mag_tensor_device_id(x)));
// ... promote x,y via mag_cast if needed ...
mag_tensor_t *in[2] = {prom_x, prom_y};
mag_try(mag_check_dtype_and_device_compat(err, op, in, 0));
mag_try(mag_dispatch(err, op, flags & MAG_BINOP_INPLACE, NULL, in, 2, &result, 1));
```

And a pure metadata op (`mag_view`) never touches storage at all - it calls
`mag_as_strided` (section 3) instead of `mag_empty`, then dispatches anyway
so the autodiff tape still records that a view happened:

```513:520:refs/magnetron/magnetron/core/mag_op_stubs.c
mag_try(mag_as_strided(err, &result, x->ctx, x, rank, shape, strides, x->storage_offset));
mag_try(mag_check_dtype_and_device_compat(err, MAG_OP_VIEW, &x, 0));
mag_try(mag_dispatch(err, MAG_OP_VIEW, false, NULL, &x, 1, &result, 1));
```

This means Magnetron is not centered on kernel fusion or deferred
scheduling. The core decides semantics now, then asks the backend to run
now - and `mag_dispatch` is the single seam where "core policy" (validation,
allocation, graph recording) hands off to "backend mechanism" (`submit`).

### 5. The operator table is the control plane, and it is X-macro generated

`magnetron/magnetron/core/mag_operator.h` is more important than it looks.
Every operator in the entire framework is declared **once**, as one line in
an X-macro list (`mag_opdef`), and that single list is stitched into three
different things: the opcode enum, the dtype-support/backward-function
table, and (indirectly, via naming convention) the CPU kernel dispatch
table described in section 9.

A representative slice of the list - note the columns are name, arity-in,
arity-out, allowed-dtype-mask, op-attrs, flags, and a suffix used to build
the backward function name:

```107:109:refs/magnetron/magnetron/core/mag_operator.h
  _(ADD, 2, 1, NUMERIC, {}, MAG_OP_FLAGS_COMMON, add)__
  _(MUL, 2, 1, NUMERIC, {}, MAG_OP_FLAGS_COMMON, mul)__
```

```44:44:refs/magnetron/magnetron/core/mag_operator.h
  _(VIEW, 1, 1, ALL, {}, MAG_OP_FLAG_NONE, view)__
```

The opcode enum falls directly out of the same list:

```136:144:refs/magnetron/magnetron/core/mag_operator.h
typedef enum mag_opcode_t {
#define _(enu, in, out, dtm, opp, flags, diff) MAG_OP_##enu
  mag_opdef(_, MAG_SEP)
#undef _
  MAG_OP__NUM
} mag_opcode_t;
mag_static_assert(MAG_OP_NOP == 0);
mag_static_assert(MAG_OP_CLAMP+1 == MAG_OP__NUM);
mag_static_assert(MAG_OP__NUM <= 0xff); /* Must fit in one byte */
```

There are 101 opcodes total (`MAG_OP_NOP` = 0 through `MAG_OP_CLAMP` = 100),
of which 26 carry a non-`NULL` backward function - the rest (comparisons,
`WHERE`, `MIN`/`MAX`, most reductions in this snapshot) are currently
forward-only. The descriptor struct these entries populate:

```163:171:refs/magnetron/magnetron/core/mag_operator.h
typedef struct mag_op_traits_t {
  const char *const mnemonic;
  const uint32_t in;
  const uint32_t out;
  const mag_dtype_mask_t dtype_mask;
  const mag_op_attr_type_tag_t op_attr_types[MAG_MAX_OP_PARAMS];
  const mag_opflags_t flags;
  mag_status_t (*const backward)(mag_error_t *, mag_au_state_t *, mag_tensor_t **);
} mag_op_traits_t;
```

and `mag_operator.c` builds the whole static table from the exact same
X-macro list a second time, this time keying a designated-initializer array
by opcode and gluing on `mag_op_backward_##diff` (which resolves to
`mag_op_backward_NULL` -> `NULL` for ops with no gradient rule):

```15:22:refs/magnetron/magnetron/core/mag_operator.c
const mag_op_traits_t *mag_op_traits(mag_opcode_t op) {
  static const mag_op_traits_t infos[MAG_OP__NUM] = {
  #define _(enu, in, out, dtm, opp, flags, diff) [MAG_OP_##enu] = (mag_op_traits_t){ \
    #enu, in, out, MAG_DTYPE_MASK_##dtm, opp, flags, mag_op_backward_##diff }
    mag_opdef(_, MAG_SEP)
  };
  return infos+op;
}
```

If you want to understand "what is the framework, really?", this file (one
list, expanded three ways) is one of the best to read early - it is the
compact specification of the runtime's entire operator surface, and it is a
good illustration of a technique comtam can use on a much smaller scale once
it has more than a handful of ops: **one source list, multiple mechanical
expansions**, instead of an enum, a switch statement, and a docs table that
can silently drift out of sync with each other.

### 6. Autodiff is dynamic reverse-mode built from recorded inputs

The relevant files are:

- `magnetron/magnetron/core/mag_autodiff.h`
- `magnetron/magnetron/core/mag_autodiff.c`
- `magnetron/magnetron/core/mag_gradients.c`
- `magnetron/magnetron/core/mag_toposort.c`
- `magnetron/magnetron/core/mag_operator.h`

The tape state attached to a tensor is small and fixed-size - at most two
input references (`MAG_MAX_OP_INPUTS`), because every current op has arity
<= 2 or is a view/reduction that only needs one gradient path:

```22:30:refs/magnetron/magnetron/core/mag_autodiff.h
struct mag_au_state_t {
  MAG_RC_INJECT_HEADER;
  mag_context_t *ctx;
  mag_opcode_t op;
  mag_tensor_t *op_inputs[MAG_MAX_OP_INPUTS];
  mag_op_attr_t op_attrs[MAG_MAX_OP_PARAMS];
  mag_tensor_t *grad;
};
```

This state is lazily allocated (from the context's `au_state_slab`) the
first time it's needed, and is populated **on each op's output tensor**
inside `mag_dispatch` (section 4) - never on a separate global tape
object.

Backward starts from a scalar root and walks a topologically sorted
post-order of the recorded graph:

```75:100:refs/magnetron/magnetron/core/mag_autodiff.c
mag_status_t mag_tensor_backward(mag_error_t *err, mag_tensor_t *root) {
  mag_ctx_grad_recorder_stop(root->ctx);
  mag_topo_sort(root, &post_order);
  // reverse post_order (root-first for backward)
  for (size_t id=0; id < post_order.size; ++id) {
    mag_tensor_t *child = post_order.data[id];
    if (!child->au_state->grad)
      /* seed with ones for the root, else must already be populated */;
    if (child->au_state->op == MAG_OP_NOP) continue;
    stat = meta->backward(err, child->au_state, grads);
    /* accumulate into inputs' .grad via mag_add, or patch if first write */
  }
  mag_ctx_grad_recorder_start(root->ctx);
  return stat;
}
```

Gradient recording is explicitly turned off for the duration of backward
(`mag_ctx_grad_recorder_stop`/`_start`) - backward passes do not themselves
get taped, avoiding an accidental second-order tape unless a user asks for
one on purpose.

The graph traversal (`mag_toposort.c`) is an iterative (not recursive) DFS
over `op_inputs`, using an explicit stack of `(tensor, next_child_idx)`
frames so it never blows the C call stack on a deep graph, and only follows
edges into tensors that actually `require_grad`:

```78:96:refs/magnetron/magnetron/core/mag_toposort.c
void mag_topo_sort(mag_tensor_t *root, mag_topo_set_t *out_sorted) {
  if (!(root->flags & MAG_TFLAG_REQUIRES_GRAD)) return;
  mag_topo_stack_push(&stack, root);
  while (stack.len) {
    mag_topo_stack_record_t *top = mag_topo_stack_peek(&stack);
    uint32_t num_children = mag_op_traits(au->op)->in;
    if (top->next_child_idx >= num_children) {
      mag_topo_stack_pop(&stack);
      mag_topo_set_push(out_sorted, top_t);  /* post-order */
      continue;
    }
    mag_tensor_t *child = au->op_inputs[top->next_child_idx++];
    if (child && child->flags & MAG_TFLAG_REQUIRES_GRAD && !visited)
      mag_topo_stack_push(&stack, child);
  }
}
```

This is a classic dynamic autograd tape, closer to PyTorch-style eager
autograd than to a lazy compiled graph.

The detail to pay attention to is where graph edges live:

- not in a separate graph object
- not in a global tape
- on each tensor's autodiff state

That keeps the system small and inspectable, and it is directly comparable
to how comtam's own autograd (Module 6) should probably work: attach
`(op, inputs, attrs, grad)` to the *output* tensor at dispatch time, and let
backward be "topo sort backward from a scalar, call each op's registered
backward function, accumulate."

Gradient rules themselves live in `mag_gradients.c`, one function per
differentiable op, matching the `backward` field wired up in section 5. Add
and Mul are worth reading back to back because they show how broadcasting
is undone on the way back (`mag_repeat_back` sums the gradient back down to
the smaller input's original shape when the forward op broadcast it):

```276:286:refs/magnetron/magnetron/core/mag_gradients.c
mag_status_t mag_op_backward_add(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->op_inputs[0];
  mag_tensor_t *y = node->op_inputs[1];
  if (x->flags & MAG_TFLAG_REQUIRES_GRAD)
    mag_try(mag_clone(err, grads, node->grad));
  if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
    if (!mag_tensor_is_shape_eq(x, y)) mag_try(mag_repeat_back(err, &grad, node->grad, y));
    else mag_try(mag_clone(err, &grad, node->grad));
    grads[1] = grad;
  }
  return MAG_STATUS_OK;
}
```

```317:327:refs/magnetron/magnetron/core/mag_gradients.c
mag_status_t mag_op_backward_mul(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  if (x->flags & MAG_TFLAG_REQUIRES_GRAD)
    mag_try(mag_mul(err, grads, node->grad, y));
  if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
    mag_try(mag_mul(err, &xg, x, node->grad));
    if (!mag_tensor_is_shape_eq(x, y)) mag_try(mag_repeat_back(err, &xg, pxg, y));
    grads[1] = xg;
  }
  return MAG_STATUS_OK;
}
```

Every backward function has the exact same signature
(`mag_status_t(mag_error_t*, mag_au_state_t*, mag_tensor_t**)`), which is
what lets `mag_op_traits_t::backward` be a single, uniform function pointer
type across all 26 differentiable ops.

### 7. Backends are shared libraries behind a small, explicit ABI

Study:

- `magnetron/magnetron/core/mag_backend.h`
- `magnetron/magnetron/core/mag_backend.c`
- `magnetron/magnetron/CMakeLists.txt`

There are two vtables, not one - a per-backend-module vtable and a
per-device vtable - and both are small enough to read in full. The backend
module vtable answers "what devices does this module expose":

```83:93:refs/magnetron/magnetron/core/mag_backend.h
struct mag_backend_t {
  void *impl;
  bool (*init)(mag_backend_t *self, mag_context_t *ctx);
  bool (*shutdown)(mag_backend_t *self);
  uint32_t (*backend_version)(mag_backend_t *bck);
  uint32_t (*runtime_version)(mag_backend_t *bck);
  const char *(*id)(mag_backend_t *bck);
  uint32_t (*num_devices)(mag_backend_t *bck);
  uint32_t (*best_device_id)(mag_backend_t *bck);
  mag_device_t *(*get_device)(mag_backend_t *bck, uint32_t idx);
};
```

The device vtable answers "how do I actually run something on this
specific device" - and note that `submit` (the function `mag_dispatch`
calls) lives here, not on the backend module:

```69:79:refs/magnetron/magnetron/core/mag_backend.h
struct mag_device_t {
  void *impl;
  mag_context_t *ctx;
  mag_device_id_t id;
  bool is_async;
  mag_status_t (*submit)(mag_device_t *, mag_error_t *, const mag_command_t *);
  mag_status_t (*alloc_storage)(mag_device_t *, mag_error_t *, mag_storage_buffer_t **, size_t);
  void (*manual_seed)(mag_device_t *, mag_error_t *, uint64_t);
  mag_status_t (*transfer)(mag_device_t *, mag_error_t *, mag_transfer_dir_t, mag_tensor_t *, mag_tensor_t *);
  char physical_device_name[256];
};
```

The runtime loads backend modules dynamically from shared libraries such as
`magnetron_cpu` and `magnetron_cuda`. Loading is guarded by an explicit ABI
cookie check before a single vtable function pointer is ever called:

```39:58:refs/magnetron/magnetron/core/mag_backend.c
static mag_backend_module_t *mag_backend_module_load(const char *file, mag_context_t *ctx) {
  mag_dylib_t *handle = mag_dylib_open(file);
  // resolve: mag_backend_module_hook_abi_cookie, _init, _shutdown
  uint32_t abi_cookie = (*(MAG_BACKEND_SYM_FN_ABI_COOKIE*)fn_abi_cookie)();
  uint32_t curr_cookie = mag_pack_abi_cookie('M', 'A', 'G', MAG_BACKEND_MODULE_ABI_VER);
  if (abi_cookie != curr_cookie) { /* reject */ }
  mag_backend_t *backend = (*(MAG_BACKEND_SYM_FN_INIT*)fn_init)(ctx);
  // verify all vtable fn pointers non-NULL, verify runtime_version() == MAG_VERSION
  // call backend->init(backend, ctx)
}
```

The core checks:

- ABI cookie (a packed `'M','A','G',version` tag, currently version 1)
- runtime version compatibility against the core's own `MAG_VERSION`
- every required vtable function pointer is non-`NULL` before trusting the
  module at all

This is a strong architectural choice. The backend interface is not an
internal convenience layer. It is a real plugin boundary, checked
defensively at load time rather than assumed. That makes Magnetron feel
more like a small runtime platform than a single monolithic library.

Device resolution from a device id (`{type, ordinal}`) is equally small -
notice that **the C core never parses strings like `"cpu:0"`**; that parsing
lives in the Python binding layer (`prelude.cpp`'s `parse_device_id_str`),
and the core only ever sees a plain struct:

```229:234:refs/magnetron/magnetron/core/mag_backend.c
bool mag_backend_registry_get_backend_and_device_by_id(...) {
  mag_backend_t *bck = mag_backend_registry_get_backend(reg, id.type);
  if (mag_unlikely(!bck)) return false;
  if (mag_unlikely(id.device_ordinal >= (*bck->num_devices)(bck))) return false;
  mag_device_t *dvc = (*bck->get_device)(bck, id.device_ordinal);
}
```

### 8. The CPU backend is already the serious backend

The CPU backend is the mature backend right now.

Start with:

- `magnetron/magnetron/cpu/CMakeLists.txt`
- `magnetron/magnetron/cpu/mag_cpu.c`

Then notice the rest of the directory:

- architecture-specific configuration under `amd64/` and `arm64/`
- operator implementations in `mag_cpu_kernels_*.h`
- threadpool and phase-fence infrastructure
- specialization and autotuning helpers

Important design points:

- the backend picks optimized CPU kernels based on detected hardware
- work can run single-threaded or through the CPU threadpool
- storage allocation is device-owned even on CPU

This is where Magnetron differs sharply from educational runtimes that
treat CPU as a toy fallback. Here CPU is the primary systems target, and the
mechanism behind "picks optimized CPU kernels based on detected hardware"
is detailed enough to deserve its own section - see section 9 below, which
is the CPU-backend analog of MLX's kernel-folder complexity
([`MLX_KERNELS.md`](MLX_KERNELS.md)).

### 9. CPU kernel organization: one kernel body, many ISA-specific compiled copies

This is the part of Magnetron that plays the same role as MLX's
`kernels/` folder: a lot of files that look complicated but boil down to a
few repeated, mechanical layers. Where MLX multiplies kernel *source* by
(op x dtype x layout) via C++ templates and a Metal `[[host_name]]` string,
Magnetron multiplies kernel *object code* by (op x dtype x CPU
microarchitecture) via C translation-unit duplication and a plain C
function-pointer table. Same underlying problem - "how do I ship the best
implementation of an op for whatever hardware I actually end up running on"
- solved with the tools available in each language.

#### 9.1 The file layout

```text
magnetron/cpu/
|-- mag_cpu.c                        backend vtable, submit(), device init
|-- mag_cpu_kernel_data.h            mag_kernel_registry_t: the dispatch table type
|-- mag_cpu_specialization_detector.c  CPUID/hwcap probing -> pick a specialization
|-- mag_cpu_autotune.c               per-op thread count, matmul block-size heuristics
|-- mag_cpu_threadpool.c             phase-fence barrier thread pool
|-- kernels/
|   |-- mag_cpu_dispatch.h           THE shared kernel translation unit (see 9.2)
|   |-- mag_cpu_simd.h               portable SIMD vector types (mag_vf32_t, ...)
|   |-- mag_cpu_simd_functions.h     vectorized math (exp, tanh, sincos, ...)
|   |-- mag_cpu_kernels_unary.h      unary op bodies
|   |-- mag_cpu_kernels_binary.h     binary op bodies
|   |-- mag_cpu_kernels_reduction.h  reduction op bodies
|   |-- mag_cpu_kernels_matmul.h     matmul op bodies
|   `-- mag_cpu_kernels_{cast,fill,misc}.h
|-- amd64/
|   |-- amd64.cmake                  per-file -march flags -> MAG_HAVE_CPU_* macros
|   `-- mag_cpu_amd64_{core2,...,zn1..zn5}.c   21 one-line wrapper files
`-- arm64/
    |-- arm64.cmake
    `-- mag_cpu_arm64_{v82,v86,v86_crypto,v82_sve,v9_sve2}.c   5 wrapper files
```

#### 9.2 Each microarchitecture file is a two-line wrapper, not unique code

Every file under `amd64/` and `arm64/` has the same shape - two macro
`#define`s and one `#include`. No kernel logic is duplicated by hand
anywhere:

```12:15:refs/magnetron/magnetron/cpu/amd64/mag_cpu_amd64_icelake.c
#define MAG_BLAS_SPECIALIZATION mag_cpu_blas_specialization_amd64_icelake
#define MAG_BLAS_SPECIALIZATION_FEAT_REQUEST mag_cpu_blas_specialization_amd64_icelake_features

#include "../kernels/mag_cpu_dispatch.h"
```

```12:15:refs/magnetron/magnetron/cpu/amd64/mag_cpu_amd64_zn3.c
#define MAG_BLAS_SPECIALIZATION mag_cpu_blas_specialization_amd64_zn3
#define MAG_BLAS_SPECIALIZATION_FEAT_REQUEST mag_cpu_blas_specialization_amd64_zn3_features

#include "../kernels/mag_cpu_dispatch.h"
```

CMake compiles each of these wrapper files as its **own translation unit**
with a **different `-march` flag**, and only if the host compiler actually
supports that flag (`mag_register_cpu_backend`, `check_c_compiler_flag`):

```11:26:refs/magnetron/magnetron/cpu/amd64/amd64.cmake
mag_register_cpu_backend("amd64/mag_cpu_amd64_icelake.c"  "-march=icelake-client -mtune=icelake-client" "/arch:AVX512")
mag_register_cpu_backend("amd64/mag_cpu_amd64_zn3.c"      "-march=znver3 -mtune=znver3"                 "/arch:AVX2")
mag_register_cpu_backend("amd64/mag_cpu_amd64_zn4.c"      "-march=znver4 -mtune=znver4"                 "/arch:AVX512")
```

Because the same `mag_cpu_dispatch.h` (and therefore the same kernel bodies)
gets compiled once per accepted arch flag, the compiler itself generates a
distinct, differently-vectorized copy of every kernel function per
microarchitecture - `mag_add_float32` compiled under `-march=icelake-client`
uses AVX-512 instructions; the same source line compiled under
`-march=znver3` uses AVX2. There is no `#ifdef` ladder inside the kernel
bodies choosing instructions; the ISA choice is entirely a build-flag
concern, and the kernel source stays portable C.

All accepted TUs link into one shared library:

```16:22:refs/magnetron/magnetron/cpu/CMakeLists.txt
set(MAGNETRON_CPU_SOURCES ${MAGNETRON_CPU_SOURCES} ${MAG_CPU_SOURCES_ENABLED})
add_library(magnetron_cpu SHARED ${MAGNETRON_CPU_SOURCES})
if (MAG_ENABLED_CPU_MACROS)
    target_compile_definitions(magnetron_cpu PRIVATE ${MAG_ENABLED_CPU_MACROS})
endif()
```

so `magnetron_cpu.{so,dylib}` on a given machine contains, e.g., an Ice
Lake-optimized copy of every op *and* a Zen 3-optimized copy of every op
*and* a plain scalar fallback copy, all at once - the choice of which one
actually runs happens once, at process startup, not at build time.

#### 9.3 `mag_cpu_dispatch.h`: the one file every kernel body actually lives in

This header is not a dispatcher despite its name - it is the shared
compilation unit that every arch wrapper `#include`s. It pulls in every
kernel implementation header in a fixed order:

```116:124:refs/magnetron/magnetron/cpu/kernels/mag_cpu_dispatch.h
/* Order matters, do not touch */
#include "mag_cpu_crc32c.h"
#include "mag_cpu_kernels_unary.h"
#include "mag_cpu_kernels_cast.h"
#include "mag_cpu_kernels_binary.h"
#include "mag_cpu_kernels_fill.h"
#include "mag_cpu_kernels_matmul.h"
#include "mag_cpu_kernels_misc.h"
#include "mag_cpu_kernels_reduction.h"
```

builds a **static, file-local** 2D lookup table
`mag_lut_eval_kernels[MAG_OP__NUM][MAG_DTYPE__NUM]` of function pointers
(one distinct copy of this table per compiled TU, since it's `static`), and
finally exports exactly two symbols per TU, named via the two macros each
wrapper file defined:

```1149:1159:refs/magnetron/magnetron/cpu/kernels/mag_cpu_dispatch.h
void MAG_BLAS_SPECIALIZATION(mag_kernel_registry_t *registry) {
  registry->init = &mag_impl_init;
  registry->deinit = &mag_impl_deinit;
  for (int i=0; i < MAG_OP__NUM; ++i) {
    for (int j=0; j < MAG_DTYPE__NUM; ++j) {
      registry->operators[i][j] = mag_lut_eval_kernels[i][j];
    }
  }
  registry->vreg_width = &mag_vreg_width;
  registry->crc32c = &mag_crc32c;
}
```

and a matching feature-bitmask function that reports, from compiler
predefined macros, which ISA extensions **this specific compiled TU**
actually used:

```1178:1188:refs/magnetron/magnetron/cpu/kernels/mag_cpu_dispatch.h
mag_amd64_cap_bitset_t MAG_BLAS_SPECIALIZATION_FEAT_REQUEST() {
  mag_amd64_cap_bitset_t caps = 0;
#ifdef __SSE__
  caps|=mag_amd64_cap(SSE);
#endif
#ifdef __AVX2__
  caps|=mag_amd64_cap(AVX2);
#endif
#ifdef __AVX512F__
  caps|=mag_amd64_cap(AVX512_F);
#endif
  return caps;
}
```

So `MAG_BLAS_SPECIALIZATION`/`_FEAT_REQUEST` are not real function names -
they're placeholders each wrapper file renames via `#define` before
including the shared header, which is how one header produces N
differently-named, differently-compiled exported functions instead of a
duplicate-symbol link error.

#### 9.4 The dispatch table itself is a plain 2D array, not a name lookup

```40:46:refs/magnetron/magnetron/cpu/mag_cpu_kernel_data.h
typedef struct mag_kernel_registry_t {
  void (*init)(void);
  void (*deinit)(void);
  mag_status_t (*operators[MAG_OP__NUM][MAG_DTYPE__NUM])(mag_error_t *, const mag_kernel_payload_t *);
  size_t (*vreg_width)(void);
  uint32_t (*crc32c)(const void *buf, size_t nb);
} mag_kernel_registry_t;
```

Where MLX looks up a kernel by building a string and searching a Metal
library, Magnetron indexes a flat C array by `[opcode][dtype]` - no string
comparison, no hashing, at dispatch time:

```36:46:refs/magnetron/magnetron/cpu/mag_cpu_threadpool.c
mag_opcode_t op = payload->cmd->op;
mag_dtype_t dtype = mag_command_dispatch_dtype(payload->cmd);
mag_status_t (*kernel)(mag_error_t *, const mag_kernel_payload_t *) = kernels->operators[op][dtype];
mag_contract(err, ERR_MISSING_COMPUTE_KERNEL, {}, kernel != NULL, ...);
mag_status_t stat = (*kernel)(err, payload);
```

This only works because opcode and dtype are both small, dense, known-at-
compile-time enums (checked with `mag_static_assert(MAG_OP__NUM <= 0xff)`
in `mag_operator.h`) - the entire naming problem MLX solves at runtime with
strings, Magnetron solves at compile time with enum values used as array
indices.

#### 9.5 Runtime selection: CPUID/hwcap probing picks exactly one specialization, once

Host capabilities are probed once at context creation
(`mag_machine_info_probe`), via real `CPUID` leaves plus `XGETBV` on x86:

```52:57:refs/magnetron/magnetron/core/mag_cpuid.c
void mag_probe_cpu_amd64(mag_amd64_cap_bitset_t *o, uint32_t *avx10ver) {
  memset(o, 0, sizeof(*o));
  mag_cpuid(&id, 0);
  if (mag_cpuid_streq(*ebx, *ecx, *edx, "AuthenticAMD")) *o|=mag_amd64_cap(AMD);
  else if (mag_cpuid_streq(*ebx, *ecx, *edx, "GenuineIntel")) *o|=mag_amd64_cap(INTEL);
```

or `getauxval(AT_HWCAP/AT_HWCAP2)` / `IsProcessorFeaturePresent` on ARM64:

```346:354:refs/magnetron/magnetron/core/mag_cpuid.c
void mag_probe_cpu_arm64(mag_arm64_cap_bitset_t *o, int64_t *sve_width) {
  *o = MAG_ARM64_CAP_NONE;
#ifdef __linux__
  unsigned long hwcap = getauxval(AT_HWCAP);
  *o|=mag_arm64_cap(NEON);
  if (hwcap & HWCAP_ASIMDDP) *o|=mag_arm64_cap(DOTPROD);
```

`mag_cpu_specialization_detector.c` keeps a **preference-ordered** static
array per vendor (best microarchitecture first, `core2`/generic last), and
walks it once, injecting the first specialization whose required feature
bitmask is fully satisfied by the host:

```257:271:refs/magnetron/magnetron/cpu/mag_cpu_specialization_detector.c
bool mag_blas_detect_optimal_specialization(const mag_context_t *ctx, mag_kernel_registry_t *kernels) {
  const mag_cpu_specialization_t *impls = mag_get_cpu_specializations(ctx, &num_impls);
  uint64_t host_caps = mag_get_cpu_host_caps(ctx);
  for (size_t i=0; i < num_impls; ++i) {
    const mag_cpu_specialization_t *spec = impls+i;
    uint64_t spec_caps = (*spec->get_feature_bitset)();
    bool matches = (host_caps&spec_caps) == spec_caps;
    if (matches) { (*spec->inject_kernels)(kernels); return true; }
  }
  /* fallback: scalar generic kernels */
}
```

This runs once, during CPU device initialization, before the threadpool is
created - not per-op, not per-tensor:

```141:150:refs/magnetron/magnetron/cpu/mag_cpu.c
static mag_cpu_device_t *mag_cpu_init_device(mag_context_t *ctx, uint32_t num_threads) {
  mag_blas_detect_optimal_specialization(ctx, &dvc->kernels);
  if (num_threads > 1) {
    dvc->pool = mag_threadpool_create(ctx, num_threads, &dvc->kernels, ...);
  }
  if (*dvc->kernels.init) (*dvc->kernels.init)();
```

After this point, `dvc->kernels.operators[op][dtype]` is a fixed function
pointer for the rest of the process - there is no per-call re-selection, no
JIT, and nothing analogous to MLX's runtime kernel compilation. All the
"which variant" decision-making happens exactly once.

#### 9.6 A generic SIMD abstraction, not per-op vector intrinsics

Kernel bodies do not write raw `_mm256_add_ps`/NEON intrinsics inline.
`mag_cpu_simd.h` picks one portable vector typedef per compiled TU based on
which ISA macros the compiler defines for that `-march`:

```43:56:refs/magnetron/magnetron/cpu/kernels/mag_cpu_simd.h
#if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
  typedef float32x4_t mag_vf32_t;
#elif defined(__AVX512F__)
  typedef __m512 mag_vf32_t;
#elif defined(__AVX2__)
  typedef __m256 mag_vf32_t;
#elif defined(__SSE2__)
  typedef __m128 mag_vf32_t;
#else
  typedef float mag_vf32_t;   /* scalar fallback */
#endif
#define MAG_VF32_LANES ((int64_t)(sizeof(mag_vf32_t)/sizeof(float)))
```

and every kernel is written once against `mag_vf32_t`/`MAG_VF32_LANES` plus
a macro that generates the vector-main-loop-plus-scalar-tail pattern, which
is instantiated per op via a second macro layer:

```96:112:refs/magnetron/magnetron/cpu/kernels/mag_cpu_kernels_binary.h
#define mag_gen_bin_simd(T, TF, suffix, LOAD, STORE, name) \
  static mag_status_t MAG_HOTPROC mag_##name##_##TF(...) { \
    int64_t chunk = (total+tc-1)/tc; \
    int64_t ra = ti*chunk, rb = mag_xmin(ra+chunk,total); \
    int64_t i=ra; \
    for (; i+MAG_VF32_LANES <= rb; i += MAG_VF32_LANES) { \
      mag_vf32_t vx = LOAD(bx+i), vy = LOAD(by+i); \
      STORE(br+i, mag_vec_##name##_f32(vx,vy)); \
    } \
    for (; i < rb; ++i) br[i] = mag_fn_##name##_##suffix(bx[i],by[i]); \
  }
```

This is the same "template the kernel body, instantiate per (op, dtype)"
idea seen in MLX's `instantiate_kernel` macro
([`MLX_KERNELS.md`](MLX_KERNELS.md) section 5) - just one level lower in the
toolchain: MLX instantiates a C++ template into named Metal functions per
(op, dtype, layout); Magnetron instantiates a C macro into named C functions
per (op, dtype), and then relies on **separate compilation with different
flags** (not templates) to get the per-architecture variation.

Reductions, notably, are not vectorized this way in the current snapshot -
`mag_cpu_kernels_reduction.h`'s sum kernel is a scalar accumulate loop
generated by a plain macro, with no SIMD path and no thread partitioning
(the autotuner's `growth` factor for `MAG_OP_SUM` is `0`, forcing
single-threaded execution - see 9.8). That's a concrete, useful data point
if you're deciding how much reduction-kernel sophistication is worth
building for comtam right now: even a mature reference framework ships a
plain scalar reduction on CPU and reserves the parallel/vectorized
sophistication for matmul and elementwise ops instead.

#### 9.7 Parallelism: a phase-fence barrier, not work-stealing

The thread pool is not work-stealing. It's a fixed-size pool synchronized
per-op by an atomic phase counter (`mag_phase_fence_t`): the main thread
bumps the phase and wakes all workers, every worker (main thread included,
as worker 0) calls the exact same `operators[op][dtype]` lookup and runs
its assigned slice, then signals done and the main thread blocks until all
workers finish:

```18:22:refs/magnetron/magnetron/cpu/mag_cpu_phase_fence.c
void mag_phase_fence_kick(mag_phase_fence_t *fence, int32_t workers_active) {
  mag_atomic32_store(&fence->remaining, workers_active, MAG_MO_RELAXED);
  mag_atomic32_store(&fence->phase, cur+1, MAG_MO_RELEASE);
  mag_futex_wakeall(&fence->phase);
}
```

Within one kernel invocation, work is split two different ways depending on
the op: elementwise kernels use a **fixed contiguous chunk per thread**
(`chunk = ceil(total/thread_num)`, seen in the binary-op macro above);
matmul instead uses **dynamic tile stealing** via an atomic tile counter
(`mag_atomic64_fetch_add(payload->mm_next_tile, 1)`), because matmul tiles
have much more variable cost per tile than a fixed elementwise chunk does.

#### 9.8 Autotuning: a per-op heuristic, not a benchmark cache

`mag_cpu_autotune.c` does **not** benchmark kernel variants at runtime and
cache a winner - that decision (which ISA specialization to use) already
happened once in 9.5. What it tunes, on every single `submit` call, is how
many of the already-allocated worker threads should actually participate in
*this* op, based on a per-opcode growth curve:

```231:239:refs/magnetron/magnetron/cpu/mag_cpu_autotune.c
uint32_t mag_cpu_dynamic_work_scaling(uint32_t allocated_workers, mag_opcode_t op, int64_t numel) {
  mag_op_thread_scaling_info info = mag_cpu_get_op_thread_scaling_info(op);
  if (allocated_workers <= 1 || numel < info.thread_treshold)
    return 1;
  uint32_t workers = (uint32_t)ceil(info.growth * log2((double)numel));
  return mag_xmin(allocated_workers, mag_xmax(1, workers));
}
```

For matmul specifically it goes further and analytically derives cache-tile
parameters (`MR`, `NR`, `MC`, `KC`, `NC`) from the detected L1/L2 cache
sizes and vector width (`registry->vreg_width()`) rather than sweeping
options - a small, deterministic, closed-form autotuner rather than a
search-based one. This is a good example of "no optimization before a
measurement forces it" done *correctly at framework scale*: the tuning
formula itself was presumably derived from measurement once, then baked in
as a closed-form heuristic rather than re-measured on every run.

#### 9.9 Summary: the CPU analog of a GPU kernel-name lookup

| GPU (Metal/CUDA, see `MLX_KERNELS.md`) | Magnetron CPU |
|---|---|
| One `.metal`/`.cu` template source | One `mag_cpu_dispatch.h` + kernel headers |
| Compile many named variants (dtype x layout) | Compile many `.c` translation units (one per `-march`) |
| Runtime: look up a kernel by `[[host_name]]` string in a library | Runtime: CPUID/hwcap match picks one specialization, once |
| `MTL::ComputePipelineState` cache keyed by string | `operators[opcode][dtype]` function-pointer array, keyed by enum |
| Chosen per kernel call (name built each dispatch) | Chosen once per process (injected at device init) |
| Wrong variant on wrong device = illegal instruction/crash | Guarded: only injected if `(host_caps & required) == required` |

The throughline worth keeping across both frameworks: **the host-visible
"kernel" is decoupled from its compiled representation** - a stable name
(Metal `[[host_name]]`) or a stable index (opcode/dtype pair) is the actual
interface; the underlying machine code backing it can be swapped by build
configuration or runtime detection without the calling code changing at
all.

### 10. CUDA exists, but the architecture still reads as CPU-first

Look at:

- `magnetron/magnetron/cuda/CMakeLists.txt`
- `magnetron/magnetron/cuda/mag_cuda.cu`
- `magnetron/magnetron/cuda/mag_cuda_{binary,unary,reduction,matmul,fill,misc}.cu`

The file split mirrors the CPU kernel headers almost exactly (one `.cu`/`.cuh`
pair per op family: binary, unary, reduction, matmul, fill, misc), which is
a useful confirmation that "split kernels by op family, one file per
family" is a stable organizational choice independent of backend. The CUDA
layer is real, but in this repo snapshot it is not yet the dominant
execution story - even the README describes CUDA as still being completed
and stabilized, and unlike the CPU backend there is no multi-microarchitecture
specialization step (CUDA relies on `nvcc`'s own architecture flags and PTX
JIT instead of the CPUID-style runtime dispatch in section 9).

So if your goal is to learn the architecture, read CPU first. The CPU
backend explains the runtime more clearly, and its kernel-organization
machinery (section 9) is the most sophisticated single piece of the whole
codebase.

### 11. Snapshots are part of the runtime, not an afterthought

Study `magnetron/magnetron/core/mag_snapshot.c`.

This file defines the native `.mag` format with sections for:

- file header
- string pool
- metadata map
- tensor descriptors
- tensor buffers

The important idea is not just serialization. It is zero-copy or
memory-mapped loading as a first-class systems feature.

That is a major clue about Magnetron's ambition: it wants to be useful for
real inference workloads, not only for toy training loops.

### 12. Python `nn` and `optim` are intentionally thin

Read:

- `magnetron/python/magnetron/__init__.py`
- `magnetron/python/magnetron/nn/module.py`
- `magnetron/python/magnetron/nn/layers.py`
- `magnetron/python/magnetron/nn/loss.py`
- `magnetron/python/magnetron/optim.py`

The pattern is simple:

- `Tensor` comes from the native bindings
- `Module`, `Parameter`, layers, losses, and optimizers are written in Python
- these higher-level pieces mostly compose tensor ops rather than hiding them

This is useful to study because it shows a clean split:

- native code for the runtime-critical path
- Python for ergonomics and model composition

---

## The Best Reading Order

If you try to read Magnetron front to back, you will drown in details. Use this order instead.

### Pass 1: Understand the product

Read these first:

- `magnetron/README.md`
- `magnetron/docs/Magnetron-Cheatsheet.md`
- `magnetron/examples/xor/main.py`
- `magnetron/examples/qwen3/README.md`

Goal:

- understand what the framework promises to users
- see both the tiny training case and the real inference case

### Pass 2: Find the Python/native boundary

Read:

- `magnetron/python/magnetron/__init__.py`
- `magnetron/magnetron/bindings/_module.cpp`
- `magnetron/magnetron/bindings/context.cpp`
- `magnetron/magnetron/bindings/tensor_class.cpp`
- `magnetron/magnetron/bindings/tensor_operators.cpp`

Goal:

- see that Python is mostly a thin skin over the native runtime
- identify what is implemented in bindings versus what is implemented in core

### Pass 3: Learn the runtime data model

Read:

- `magnetron/include/magnetron/magnetron.h`
- `magnetron/magnetron/core/mag_context.h`
- `magnetron/magnetron/core/mag_context.c`
- `magnetron/magnetron/core/mag_tensor.h`
- `magnetron/magnetron/core/mag_tensor.c`
- `magnetron/magnetron/core/mag_coords.h`
- `magnetron/magnetron/core/mag_coords.c`
- `magnetron/magnetron/core/mag_coords_iter.h`

Goal:

- understand ownership, context lifetime, storage, shape, strides, views, and tensor metadata

### Pass 4: Learn how ops really happen

Read:

- `magnetron/magnetron/core/mag_operator.h`
- `magnetron/magnetron/core/mag_operator.c`
- `magnetron/magnetron/core/mag_op_stubs.c`
- `magnetron/magnetron/core/mag_reduce_plan.h`

Goal:

- see where argument validation, broadcasting, output allocation, and dispatch happen
- read `mag_dispatch` closely enough to redraw it from memory

### Pass 5: Learn backward

Read:

- `magnetron/magnetron/core/mag_autodiff.h`
- `magnetron/magnetron/core/mag_autodiff.c`
- `magnetron/magnetron/core/mag_toposort.c`
- `magnetron/magnetron/core/mag_gradients.c`

Goal:

- understand how Magnetron records parents during forward
- understand how it reconstructs execution order for reverse-mode autodiff
- trace `add` and `mul`'s backward rules by hand for a 3-node graph

### Pass 6: Learn the backend boundary and the CPU kernel machinery

Read:

- `magnetron/magnetron/core/mag_backend.h`
- `magnetron/magnetron/core/mag_backend.c`
- `magnetron/magnetron/cpu/mag_cpu.c`
- `magnetron/magnetron/cpu/mag_cpu_kernel_data.h`
- `magnetron/magnetron/cpu/mag_cpu_specialization_detector.c`
- one `amd64/*.c` wrapper file and one `arm64/*.c` wrapper file (they are two
  lines each; read them for real)
- `magnetron/magnetron/cpu/kernels/mag_cpu_dispatch.h` (skim; it's long, but
  skim the top for includes, the bottom for the two exported functions)
- `magnetron/magnetron/cpu/kernels/mag_cpu_simd.h`
- `magnetron/magnetron/cpu/mag_cpu_autotune.c`
- `magnetron/magnetron/cpu/mag_cpu_threadpool.c`
- `magnetron/magnetron/cpu/mag_cpu_phase_fence.c`
- `magnetron/magnetron/cuda/CMakeLists.txt` (skim only, for contrast)

Goal:

- understand how the runtime treats hardware support as pluggable modules
  at the backend level (section 7), and as pluggable *compiled kernel sets*
  at the CPU-microarchitecture level (section 9)
- see why CPU is the best backend to learn first, and why its kernel folder
  is worth studying as closely as MLX's

### Pass 7: Learn persistence and real workloads

Read:

- `magnetron/magnetron/core/mag_snapshot.c`
- `magnetron/examples/ae/main.py`
- `magnetron/examples/gpt2/main.py`
- `magnetron/examples/qwen3/main.py`

Goal:

- connect the runtime design to actual training and inference workflows

### Pass 8: Turn on runtime visibility

Read:

- `magnetron/docs/Environment Variables.md`

Then use:

```bash
MAG_LOG_LEVEL=info python magnetron/examples/xor/main.py
```

Goal:

- make backend loading, machine detection, and runtime startup more visible while you trace the code

---

## Concrete Files Worth Studying Closely

If you only read a dozen files, read these:

1. `magnetron/include/magnetron/magnetron.h`
2. `magnetron/magnetron/core/mag_context.c`
3. `magnetron/magnetron/core/mag_tensor.c`
4. `magnetron/magnetron/core/mag_operator.h`
5. `magnetron/magnetron/core/mag_op_stubs.c` (`mag_dispatch` above all else)
6. `magnetron/magnetron/core/mag_autodiff.c`
7. `magnetron/magnetron/core/mag_gradients.c`
8. `magnetron/magnetron/core/mag_backend.c`
9. `magnetron/magnetron/cpu/mag_cpu.c`
10. `magnetron/magnetron/cpu/mag_cpu_specialization_detector.c`
11. `magnetron/magnetron/cpu/kernels/mag_cpu_dispatch.h` (top and bottom)
12. `magnetron/magnetron/core/mag_snapshot.c`

That set gives you the public surface, runtime ownership model, operator
path, backward path, backend boundary, CPU kernel multi-versioning, and
persistence story.

---

## Questions To Ask While Reading

Use these as prompts while you study:

1. Where does Magnetron choose runtime simplicity over peak flexibility?
2. Which parts of the framework are explicitly runtime concerns rather than compiler concerns?
3. Why does attaching autodiff state to tensors keep the implementation small?
4. What would break if the backend boundary were not a shared-library ABI?
5. Why does the CPU backend choose "compile N times with different flags,
   pick one at startup" instead of MLX's "compile one generic kernel,
   specialize the *name* at runtime"? What would it cost Magnetron to adopt
   MLX's approach on CPU, and what would it cost MLX to adopt Magnetron's
   approach on GPU?
6. Why is the CPU kernel dispatch table a dense 2D array indexed by enum,
   while MLX's GPU kernel lookup is a string-keyed hash map? What property
   of `MAG_OP__NUM`/`MAG_DTYPE__NUM` makes the array approach viable?
7. Why is `mag_dispatch` a single private function that every op funnels
   through, rather than each op stub doing its own autodiff-recording and
   submit logic inline?

---

## Practical Study Plan

If you want a concrete sequence for a few evenings of study:

### Session 1

- Read the README and cheatsheet.
- Run through the XOR example.
- Read the Python bindings for `Tensor`.

### Session 2

- Read `mag_context` and `mag_tensor`.
- Trace how a tensor gets storage and how a view keeps aliasing metadata.

### Session 3

- Read `mag_op_stubs.c` carefully, centered on `mag_dispatch`.
- Trace `view`, `reshape`, `where`, and `matmul` end to end.

### Session 4

- Read `mag_autodiff.c`, `mag_toposort.c`, and `mag_gradients.c`.
- Manually trace one backward pass for a small expression.

### Session 5

- Read `mag_backend.c` and `cpu/mag_cpu.c`.
- Understand how commands cross the backend boundary.

### Session 6

- Read `mag_cpu_specialization_detector.c`, one `amd64/*.c` and one
  `arm64/*.c` wrapper, and skim `kernels/mag_cpu_dispatch.h`.
- Draw the "build time: N translation units -> runtime: one CPUID probe ->
  one injected function-pointer table" diagram from memory.

### Session 7

- Read `mag_snapshot.c`.
- Then inspect `examples/qwen3` to see how the runtime supports a more
  realistic inference flow.

---

## Final Takeaway

The best way to think about Magnetron is this:

It is not "tinygrad but in C".

It is a compact systems-oriented ML runtime where the main abstraction is an explicit tensor runtime with:

- concrete storage ownership
- explicit views and stride solving
- operator dispatch through a backend ABI
- dynamic reverse-mode autodiff
- a CPU backend that treats "pick the right compiled kernel for this exact
  CPU" as a first-class, load-time problem rather than an afterthought
- built-in model snapshot support

Compare its CPU kernel machinery (section 9) against MLX's GPU kernel
machinery ([`MLX_KERNELS.md`](MLX_KERNELS.md)) side by side: both frameworks
solve "one op, many hardware-specific implementations" by decoupling a
stable dispatch key (opcode+dtype enum vs. `[[host_name]]` string) from a
swappable compiled implementation, chosen once (Magnetron, at CPU device
init) or lazily (MLX, on first GPU use). That's the pattern worth carrying
into `comtam`, at whatever scale is appropriate for the current module - not
either framework's literal mechanism.
