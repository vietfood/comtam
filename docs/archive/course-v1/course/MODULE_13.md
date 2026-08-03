## Module 13: Measurement-Driven Performance

Establish reliable benchmarks, identify the dominant cost, and keep only changes
that improve the supported workload without weakening correctness.

## Why Performance Comes Here

The runtime now has stable semantics, sustained training, async lifetime rules,
and persistence. Optimizing earlier would measure a moving target. This module
teaches experimental discipline: define the workload and synchronization
boundary, measure distributions, change one mechanism, and rerun every semantic
gate.

## Module Contract

**Prerequisites:** Modules 10-12 pass and at least one supported workload has a
recorded correctness and timing baseline.

**You will produce:** a reproducible benchmark harness, per-hardware baselines,
a ranked bottleneck report, one controlled optimization experiment, and guarded
performance tracking.

**Supported scope:** end-to-end operations and training steps on documented Apple
hardware. GPU counters are used only when whole-operation timing cannot explain
the result.

**Not required:** winning against MLX/PyTorch, fusion, a graph compiler, or
optimizing every kernel.

## Assignment 13.1: Build A Reproducible Harness ⭐⭐

**Task:** benchmark representative elementwise chains, reductions, matmul shapes,
host transfer, and the Module 8 training step.

Every result records hardware, OS, build type, problem shape, warm-up, sample
count, synchronization boundary, median, p90, and dispersion. Prevent dead-code
elimination or omitted synchronization from timing work that never completes.

## Assignment 13.2: Separate Cost Categories ⭐⭐⭐

**Task:** distinguish host allocation, command encoding/submission, GPU wait,
kernel execution, readback, and view/materialization costs using the narrowest
available measurement.

Rank the top costs by their contribution to the supported workload. Do not select
an optimization based on kernel aesthetics.

## Assignment 13.3: Run One Controlled Experiment ⭐⭐⭐

**Task:** choose the top evidenced cost and change one mechanism. Candidates
include tree reduction, tiled matmul, fewer synchronization points, device-side
materialization, or bounded buffer reuse.

Before editing, write:

```text
baseline and variance
mechanism believed responsible
expected direction and minimum meaningful improvement
correctness/lifetime risks
benchmark and test commands that decide whether to keep it
```

Keep the change only if repeated measurements exceed noise and all conformance,
gradient, training, persistence, and stress checks still pass. A carefully
rejected experiment satisfies this assignment.

## Assignment 13.4: Add Performance Regression Tracking ⭐⭐

**Task:** store benchmark results as artifacts and compare like-for-like hardware
over time. Use generous, evidence-based thresholds; shared CI machines may be too
noisy for hard microsecond assertions.

Correctness CI must not become flaky because of performance variance. Prefer a
dedicated benchmark job or report-only warning until hardware is controlled.

## Assignment 13.5: Document The Next Bottleneck ⭐

**Task:** report what moved after the experiment. Optimization changes the cost
distribution, so the original second-place cost may now dominate. Rank future
work without implementing it in this module.

## Module 13 Checklist

- [ ] 13.1 Reproducible harness records environment and timing distribution.
- [ ] 13.2 Supported-workload costs are separated and ranked.
- [ ] 13.3 One optimization hypothesis is objectively kept or rejected.
- [ ] 13.4 Performance history avoids flaky correctness gates.
- [ ] 13.5 The post-experiment bottleneck report ranks future work.

## Exit Criteria

You are ready for Module 14 when:

1. Benchmark methodology is reproducible and includes synchronization correctly.
2. Bottlenecks are ranked by measured contribution to a supported workload.
3. One optimization hypothesis was tested in isolation and objectively kept or
   rejected.
4. Full correctness and lifetime gates were rerun after the experiment.
5. Performance history can distinguish regression from ordinary measurement
   noise on documented hardware.
