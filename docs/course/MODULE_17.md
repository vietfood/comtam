## Module 17: Transformer Operator Foundations

Add the tensor semantics needed by one fixed tiny GPT-2 architecture, using that
model to constrain batching, indexing, masking, normalization, and attention.

## Module Contract

**Prerequisites:** Module 16 and the CNN capstone pass.

**You will produce:** a fixed tiny GPT-2 specification and reference fixture,
batched matmul, embedding/gather, stable softmax, layer normalization, causal
masking support, required shape operations, native autograd rules, Python
bindings, and conformance tests.

**Supported scope:** float32 computation, fixed single-device eager execution,
bounded batch/sequence sizes, and the minimum indexing representation required
by the model.

**Not required:** fused attention, flash attention, KV caching, mixed precision,
arbitrary transformer variants, tokenizer implementations, or real-checkpoint
loading.

## Assignment 17.1: Freeze Tiny GPT-2 Semantics ⭐⭐

Specify dimensions, parameter names/layouts, normalization epsilon, activation,
mask convention, positional embeddings, tied weights, loss, and fixed tokens.
Capture reference intermediates, loss, gradients, and one optimizer update.

## Assignment 17.2: Add Batched Matmul And Shape Operations ⭐⭐⭐

Support only the batch broadcasting and reshape/transpose cases used by
attention. Test non-contiguous inputs, invalid dimensions, forward values, and
all differentiable inputs against independent oracles.

## Assignment 17.3: Add Embedding/Gather ⭐⭐⭐

Define index dtype, bounds behavior, repeated-index gradient accumulation, and
persistence/Python semantics. If this earns an integer dtype, implement its
storage, transfer, validation, operator, serialization, and binding contracts
together rather than treating it as a special pointer type.

## Assignment 17.4: Add Stable Softmax And Causal Masking ⭐⭐⭐

Implement numerically stable softmax on the required axes and a documented
masking representation. Test large-magnitude inputs, fully and partially masked
rows where supported, reference values, and numerical gradients.

## Assignment 17.5: Add Layer Normalization And Required Activation ⭐⭐⭐

Match the fixed model's axes, epsilon, affine parameters, and activation
formula. Compare forward intermediates and input/parameter gradients on tiny
cases, including nearly constant inputs.

## Assignment 17.6: Match A Complete Tiny Block ⭐⭐⭐

Load reference parameters and compare every boundary through embedding,
attention, residual paths, normalization, MLP, logits, loss, backward, and one
optimizer step. Record tolerances per boundary so cancellation does not hide an
early divergence.

## Module 17 Checklist

- [ ] 17.1 Tiny GPT-2 architecture and reference semantics are frozen.
- [ ] 17.2 Batched matmul and required view operations pass gradient tests.
- [ ] 17.3 Indexing has complete dtype, bounds, gradient, persistence, and Python rules.
- [ ] 17.4 Stable softmax and masking match the reference on numerical edges.
- [ ] 17.5 Normalization and activation forward/backward match the reference.
- [ ] 17.6 A full tiny block matches intermediates, gradients, and one update.

## Exit Criteria

You are ready for tiny GPT-2 training when the complete fixed model agrees with
the reference at named forward and backward boundaries, every new autograd rule
has numerical evidence, invalid indices/shapes/masks fail deterministically,
and repeated steps retain bounded graph and resource state.
