## Module 16: CNN Operator Foundations

Add the smallest tested operator surface required by the fixed CNN capstone,
without turning comtam into a general convolution library.

## Why The Workload Comes First

A CNN supplies concrete shapes, layouts, gradients, and performance pressure.
Those constraints decide which padding, convolution, pooling, and batching
semantics comtam must support. Designing a broad image API first would add
untested flexibility and obscure the actual training gate.

## Module Contract

**Prerequisites:** Modules 1-15 pass and the MLP capstone has passed its native
and Python gates.

**You will produce:** a fixed CNN architecture and reference fixture, its
missing forward operations, native autograd rules, Python exposure, conformance
tests, and a measured implementation baseline.

**Supported scope:** float32, single Apple GPU, one explicit context, explicit
NCHW or NHWC layout chosen and documented, and only the stride/padding/dilation
cases required by the capstone.

**Not required:** grouped/depthwise/transposed convolution, arbitrary image IO,
mixed precision, a general vision package, or every layout and convolution
parameter combination.

## Assignment 16.1: Freeze The CNN And Semantics ⭐⭐

Choose the exact MNIST CNN before implementing operators. Record tensor shapes
at every boundary, data layout, padding convention, parameter layout, pooling
rules, loss, optimizer, and accuracy target. Export fixed inputs, parameters,
intermediate activations, loss, and gradients from an independent reference.

## Assignment 16.2: Add Missing Scalar And Activation Ops ⭐⭐

Add only activation or elementwise operations used by the fixed network. Each
operation needs supported-semantics documentation, independent forward values,
layout and edge-case coverage, a numerical gradient test when differentiable,
and matching Python behavior.

## Assignment 16.3: Implement Narrow `conv2d` ⭐⭐⭐

Implement the documented input and weight layouts with the smallest required
stride and padding surface. A direct kernel or explicit im2col plus existing
matmul is acceptable; record materialization and synchronization costs rather
than guessing which approach is faster.

Test output shapes and values against the reference across non-square kernels,
boundary padding, batch/channel counts, and invalid configurations. Test input,
weight, and bias gradients numerically on tiny cases.

## Assignment 16.4: Implement Narrow Pooling ⭐⭐⭐

Implement only the selected pooling operation and define window, stride,
padding, tie, and empty-window behavior. Compare forward values and gradients
against the reference, including overlapping windows and boundary cases.

## Assignment 16.5: Expose Composition Through Native And Python APIs ⭐⭐

Add the minimum module wrappers needed by the fixed CNN. They must preserve the
existing context, parameter identity/naming, state dictionary, exception, and
autograd contracts. Python wrappers delegate to native implementations.

## Assignment 16.6: Establish The CNN Baseline ⭐⭐

Run bounded native and Python training smoke tests, then measure the complete
step with correct synchronization. Rank convolution, pooling, materialization,
matmul, and host overhead before keeping any optimization.

## Module 16 Checklist

- [ ] 16.1 The fixed architecture and independent fixture define required semantics.
- [ ] 16.2 Every added activation/elementwise op has forward and gradient evidence.
- [ ] 16.3 Narrow `conv2d` forward and all required gradients match the oracle.
- [ ] 16.4 Pooling semantics and gradients cover windows, ties, and boundaries.
- [ ] 16.5 Native and Python composition preserve identity and persistence rules.
- [ ] 16.6 A synchronized CNN training baseline ranks measured costs.

## Exit Criteria

You are ready for the CNN capstone when every required operation passes native
and Python conformance, tiny numerical gradient checks cover all new autograd
rules, the fixed CNN matches reference intermediates and one-step updates, and
the measured training smoke has bounded memory and graph lifetime.
