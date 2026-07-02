# MLX Reduction Kernels vs a Naive Metal Reduction

`comtam/kernels/reduction.metal` currently has a half-written `reduce_sum`
plus a leftover CUDA-flavored scratch kernel (`reduce_strided`, which uses
`threadIdx.x`/`__shared__`/`__syncthreads` - none of which exist in Metal;
the Metal equivalents are `thread_position_in_threadgroup`, `threadgroup`,
and `threadgroup_barrier`). That scratch kernel is the classic "CUDA 101"
block-level tree reduction. This note exists to show what a production Metal
framework (MLX) does differently, and why, so the module 5 assignment can be
finished with real design choices instead of a direct CUDA-to-Metal
transliteration.

This is not a "go implement all of this" note. It is a "here is the design
space, here is what MLX picked and why" note, so the actual implementation
choice for comtam stays a deliberate one.

---

## 1. The Naive Version (what's sketched in `reduction.metal` right now)

```text
load one element per thread into threadgroup memory
threadgroup_barrier
for stride = blockDim/2; stride > 0; stride /= 2:
    if tid < stride: sdata[tid] += sdata[tid + stride]
    threadgroup_barrier
if tid == 0: write sdata[0] to output
```

This is correct and a completely reasonable thing to implement once, by hand,
to feel the cost of `threadgroup_barrier` and shared-memory bank access. It
has three real limitations worth knowing about explicitly, rather than
discovering by accident:

1. **One block only.** It reduces one threadgroup's worth of data
   (`blockDim` elements) to one value. For an array bigger than one
   threadgroup, you need either a second pass or atomics to combine partial
   results across threadgroups - the naive version alone does not generalize.
2. **`log2(blockDim)` barriers.** Each halving step is a full
   `threadgroup_barrier(mem_flags::mem_threadgroup)`, which stalls every
   thread in the group. For a 256-wide threadgroup that's 8 barriers to
   combine 256 values.
3. **Ignores simdgroups.** Apple GPUs execute in 32-wide simdgroups (Metal's
   warp-equivalent) with hardware shuffle/reduce instructions that need no
   barrier at all *within* a simdgroup. The tree-in-shared-memory pattern
   never uses that hardware.

None of these make the naive version wrong. They are exactly the three axes
MLX optimizes, in order of how much they matter.

---

## 2. What MLX Does Instead: Simdgroup-First, Then One More Simdgroup Reduce

MLX's core cross-thread reduction primitive is `threadgroup_reduce` in
`kernels/reduction/reduce_row.h`. Read it as "two simdgroup reductions
instead of a log-depth tree":

```129:161:refs/mlx/mlx/backend/metal/kernels/reduction/reduce_row.h
METAL_FUNC void threadgroup_reduce(
    thread U totals[N_WRITES],
    threadgroup U* shared_vals,
  ...
) {
  Op op;

  // Simdgroup first
  for (int i = 0; i < N_WRITES; i++) {
    totals[i] = op.simd_reduce(totals[i]);
  }

  // Across simdgroups
  if (simd_per_group > 1) {
    if (simd_lane_id == 0) {
      for (int i = 0; i < N_WRITES; i++) {
        shared_vals[simd_group_id * N_WRITES + i] = totals[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    U values[N_WRITES];
    for (int i = 0; i < N_WRITES; i++) {
      values[i] = (lid.x < simd_per_group) ? shared_vals[lid.x * N_WRITES + i]
                                           : op.init;
    }

    for (int i = 0; i < N_WRITES; i++) {
      totals[i] = op.simd_reduce(values[i]);
    }
  }
}
```

Walk through it for a 256-thread threadgroup (= 8 simdgroups of 32 lanes):

1. **Phase 1 - free, no barrier.** Every lane calls `op.simd_reduce(total)`.
   For `Sum` this compiles to `simd_sum(val)`, a hardware instruction that
   combines all 32 lanes' values with no shared memory and no barrier at all.
   After this, lane 0 of each simdgroup holds that simdgroup's partial sum
   (mathematically every lane holds it, MLX just only uses lane 0's copy).
2. **Phase 2 - one barrier.** Lane 0 of each of the 8 simdgroups writes its
   partial into `shared_vals[simd_group_id]` (8 slots used out of a
   `threadgroup U shared_vals[32]` array sized for the worst case). One
   `threadgroup_barrier` makes those 8 writes visible to everyone.
3. **Phase 3 - free again.** The first 8 lanes (`lid.x < simd_per_group`)
   load one partial each (other lanes load `op.init`, the identity element,
   so they don't corrupt the result); then **all** lanes call `simd_reduce`
   again. Since only the first `simd_per_group` (<=32) lanes have real data
   and the rest have the identity, this second `simd_reduce` folds all 8
   partials into the final answer in one more hardware instruction, again
   with no barrier.

Total: **one barrier**, not eight. And the actual combining work happens via
hardware simdgroup reduction, not via threads reading/writing each other's
shared-memory slots in a loop.

This works because Apple GPU threadgroups are capped at 1024 threads = 32
simdgroups, so "one slot per simdgroup" always fits in a tiny fixed-size
`shared_vals[32]` array, and folding <=32 partial values is itself exactly
one simdgroup's job.

### The 8-byte-type exception

For types where `simd_sum`/`simd_max` don't exist in hardware (8-byte types
like `int64`/`complex64`), MLX falls back to a manual `simd_shuffle_down`
halving loop **within the simdgroup only** (5 steps for 32 lanes, still no
barrier since shuffles are intra-simdgroup):

```8:20:refs/mlx/mlx/backend/metal/kernels/reduction/ops.h
#define DEFINE_SIMD_REDUCE()                                             \
  template <typename T, metal::enable_if_t<sizeof(T) < 8, bool> = true>  \
  T simd_reduce(T val) {                                                 \
    return simd_reduce_impl(val);                                        \
  }                                                                      \
                                                                         \
  template <typename T, metal::enable_if_t<sizeof(T) == 8, bool> = true> \
  T simd_reduce(T val) {                                                 \
    for (short i = simd_size / 2; i > 0; i /= 2) {                       \
      val = operator()(val, simd_shuffle_down(val, i));                  \
    }                                                                    \
    return val;                                                          \
  }
```

The lesson: the "halving loop" idea from the naive tutorial is not wrong, it
is just applied one level down (within one 32-lane simdgroup via register
shuffles) instead of across the whole threadgroup via shared memory and
barriers.

---

## 3. The Op Is a Template Parameter, Not a Copy-Pasted Kernel

MLX writes exactly one reduction kernel body per shape-case (all/row/column)
and reuses it for `sum`, `prod`, `max`, `min`, `and`, `or` by making the
combine operation a template parameter with a uniform interface:

```118:137:refs/mlx/mlx/backend/metal/kernels/reduction/ops.h
template <typename U>
struct Sum {
  DEFINE_SIMD_REDUCE()

  template <typename T>
  T simd_reduce_impl(T val) {
    return simd_sum(val);
  }

  static constexpr constant U init = U(0);

  template <typename T>
  void atomic_update(device mlx_atomic<T>* out, T val, size_t offset = 0) {
    mlx_atomic_fetch_add_explicit(out, val, offset);
  }

  U operator()(U a, U b) {
    return a + b;
  }
};
```

Every `Op` struct provides exactly three things: an identity element
(`init`), a binary combine (`operator()`), and a simdgroup reduce
(`simd_reduce`). The kernel body never mentions `+` or `sum` directly - it
only ever calls `op(a, b)`, `op.init`, and `op.simd_reduce(val)`. This is the
same "op as a tiny functor plugged into a generic kernel" pattern comtam
could use to avoid writing `reduce_sum`, `reduce_max`, `reduce_prod` as three
near-identical Metal functions - one templated kernel plus small `Sum`/`Max`/
`Prod` structs gets all three for free, the same way
`comtam/kernels/binary.metal`'s single `add`/`sub`/`mul`/`div` bodies are
already nearly identical modulo one operator.

---

## 4. Multiple Blocks: Two-Pass, Never Atomics

The naive kernel only handles one threadgroup's worth of data. MLX's answer
for "the array is bigger than one threadgroup can hold" is a **second kernel
launch with a small intermediate buffer**, never atomics:

```280:340:refs/mlx/mlx/backend/metal/reduce.cpp
  if (in_size <= REDUCE_N_READS * 1024) {   // <= 4096 elements
    // Single threadgroup: grid == group, one launch, done.
  } else {
    // TWO-PASS reduction
    // Pass 1: n_rows independent threadgroups, each writes ONE partial
    //   <= 64MB input: n_rows = 128
    //   >  64MB input: n_rows = 4096
    // Pass 2: reduce the n_rows partials -> final scalar (same kernel again)
  }
```

Concretely: pass 1 dispatches a grid of `(threadgroup_size, n_rows, 1)`, so
threadgroup `y` reduces its own disjoint `in_size / n_rows` slice of the
input into `out[y]`. No thread ever touches another threadgroup's memory, so
there is nothing to synchronize across threadgroups - by construction, cross
threadgroup communication is entirely avoided rather than solved.

Pass 2 is *the exact same all-reduce kernel*, just invoked again on the
`n_rows`-sized intermediate buffer (which by design already fits in one
threadgroup). Row and column reductions follow the same rule: pick a grid
where **one threadgroup produces one output element** whenever possible
(so no cross-threadgroup step is needed at all), and only fall back to a
second pass when a single output needs more input than one threadgroup can
read in reasonable time (`strided_reduce_longcolumn`, `strided_reduce_2pass`
in `reduce.cpp`).

Why not atomics? `Op` structs do define `atomic_update` (used by scatter
kernels elsewhere), so MLX clearly could use atomic accumulation across
threadgroups. It doesn't, for reductions, because atomic contention from many
threadgroups hammering one output location scales badly, while a second pass
over a small, fixed-size, already-resident-in-cache intermediate buffer is
cheap and fully deterministic (no float-addition-order nondeterminism from
atomic interleaving, which matters for reproducible tests).

---

## 5. Per-Thread Unrolling: Read More Than One Element Per Thread

Before any of the above, each thread does not read one element - it reads
`REDUCE_N_READS = 4` elements per inner-loop step:

```11:13:refs/mlx/mlx/backend/metal/kernels/defines.h
static MTL_CONST constexpr int MAX_REDUCE_SPECIALIZED_DIMS = 4;
static MTL_CONST constexpr int REDUCE_N_READS = 4;
static MTL_CONST constexpr int REDUCE_N_WRITES = 4;
```

This is a memory-bandwidth optimization orthogonal to the barrier-count
optimization above: a thread that only ever touches one 4-byte float per
barrier round is latency-bound, not bandwidth-bound. Reading 4 elements per
thread per loop iteration (with an explicit tail branch for sizes not
divisible by `threadgroup_size * 4`) means fewer total threadgroup-barrier
rounds are needed to consume the same input, and memory accesses coalesce
better. This is a "do this after correctness, and only if a measurement asks
for it" optimization per the course's non-goals - but worth recognizing when
you see it, since it's easy to mistake for part of the correctness logic.

---

## 6. Row vs Column: Reduction Axis Location Changes Everything

MLX classifies every reduction into one of three geometric cases before
picking a kernel (`get_reduction_plan`, `backend/common/reduce_utils.cpp`):

- **All-reduce**: reducing every axis, input fully contiguous -> whole array
  to one scalar (`reduce_all.h`, described above).
- **Row reduce**: the axis being reduced is the fastest-moving one (stride
  1). Each output element corresponds to one contiguous *run* of input
  memory. Fast: sequential reads.
- **Column/strided reduce**: the axis being reduced has stride > 1 (it's not
  the innermost axis). Each output element's contributing elements are
  scattered `stride` bytes apart in memory. Slower: strided reads, and
  amenable to a completely different tiling strategy (`reduce_col.h` tiles a
  `BM x BN` block of the *output*, with each thread accumulating a column of
  inputs across `BM` steps).

This distinction matters more for comtam than the simdgroup trick does in the
short term: `comtam/core/view.h`'s stride/offset machinery already knows
which axis has stride 1. Reducing along the last (contiguous) axis and
reducing along an arbitrary axis are genuinely different memory-access
patterns and arguably deserve different kernels (or at least different
code paths) rather than one kernel that always walks strides generically -
exactly the "classify layout before you dispatch a kernel" idea from
[`MLX_KERNELS.md`](MLX_KERNELS.md) section 5, applied to reductions instead
of binary ops.

---

## 7. A Concrete, Scoped Recommendation for Module 5

Given the course's explicit "no optimization before a failing performance
measurement" rule, the reasonable reading of this note for the current
assignment is:

1. **Fix the CUDA/Metal mixup first.** `reduce_strided` isn't valid Metal at
   all (`__global__`, `threadIdx.x`, `__shared__`, `__syncthreads` are CUDA
   syntax) - it should either become real Metal or be deleted; keeping it
   around as dead pseudo-CUDA will not compile and does not help as a
   reference.
2. **Get one correct kernel first**: single threadgroup, threadgroup-memory
   halving tree (what's already sketched in `reduce_sum`), reducing one
   threadgroup's worth of a contiguous last axis to one output. Test it
   against an MLX-C oracle (`docs/note/MLX_C_ORACLE.md`) before touching
   anything else.
3. **Decide explicitly whether "array bigger than one threadgroup" is even
   in scope for this assignment.** If it is, the two-pass intermediate-buffer
   approach (section 4) is the pattern to reach for, not atomics - but check
   the actual `MODULE_5.md` assignment text for what's required before
   building it.
4. **Simdgroup intrinsics (`simd_sum`, `simd_max`, ...) are a legitimate,
   small, high-value follow-up** once the tree version is correct and
   tested - replacing the innermost few levels of the halving tree with one
   `simd_reduce` call is a small diff with a real payoff, and is a good
   candidate for "the first optimization I made after a measurement" in
   Module 9, not something to reach for before the naive version works.

The point of reading MLX here is not to reimplement its reduction library.
It's to make sure the design choices in `reduction.metal` (single-block vs
multi-block, tree vs simdgroup, one kernel per op vs templated op) are made
on purpose, with a known-good reference for what "more sophisticated" looks
like when the course eventually asks for it.
