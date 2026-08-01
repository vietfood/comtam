## Module 18: GPT-2 Checkpoint Interoperability

Build a verified Python-side bridge from a real pretrained GPT-2 SafeTensors
checkpoint into comtam's public API without embedding third-party formats or
tokenization into the C++ tensor core.

## Module Contract

**Prerequisites:** Module 17 passes and the tiny GPT-2 training gate passes.

**You will produce:** a pinned checkpoint/reference manifest, a validated
SafeTensors conversion or loading tool, an explicit parameter mapping, Python
model construction through public APIs, layerwise comparison fixtures, and a
deterministic generation acceptance test.

**Supported scope:** one named GPT-2 architecture and checkpoint revision,
float32 execution, fixed bounded token sequences, and offline fixtures suitable
for repeatable tests.

**Not required:** model downloading inside comtam, tokenizer implementation,
native C++ SafeTensors parsing, arbitrary Hugging Face model compatibility,
mixed precision, KV caching, or full-size GPT-2 training.

## Assignment 18.1: Pin The External Contract ⭐⭐

Record checkpoint repository/revision, file hashes, SafeTensors metadata,
reference library/version, tokenizer/version, architecture configuration, and
fixed token IDs. Network access must not be required for ordinary conformance
tests once fixtures are prepared.

## Assignment 18.2: Validate And Map Parameters ⭐⭐⭐

Use established Python SafeTensors tooling to reject missing, extra, duplicate,
wrong-shaped, or unsupported tensors before mutating a live model. Document
every rename, transpose, reshape, tied-weight rule, and dtype conversion. Apply
the mapping through comtam's public Python state API.

## Assignment 18.3: Compare Layerwise Inference ⭐⭐⭐

On fixed token IDs, compare embeddings, selected attention and MLP boundaries,
each block output, final normalization, and logits with the pinned reference.
Use explicit tolerances and stop at the first divergent boundary.

## Assignment 18.4: Verify Deterministic Generation ⭐⭐

Implement generation composition in Python using public tensor operations.
For greedy decoding from fixed token IDs, match the reference logits and
generated token IDs step by step. Token text is presentation evidence; token
IDs and logits are the correctness contract.

## Assignment 18.5: Measure And Bound The Workload ⭐⭐

Record peak live tensor/storage memory, checkpoint conversion time, model load
time, prefill latency, per-token latency, synchronization boundaries, and the
tested maximum sequence length. Optimize only after the dominant measured cost
is identified.

## Assignment 18.6: Rehearse The GPT-2 Product Claim ⭐⭐⭐

Install release artifacts in a clean environment, apply the pinned checkpoint
mapping without internal extension symbols, run layerwise and generation
acceptance fixtures, and archive hashes, version/build reports, logs, and
benchmark results.

## Module 18 Checklist

- [ ] 18.1 Checkpoint, reference, tokenizer, configuration, tokens, and hashes are pinned.
- [ ] 18.2 Parameter validation and mapping are complete and atomic on failure.
- [ ] 18.3 Named intermediate activations and logits match explicit tolerances.
- [ ] 18.4 Greedy generation matches reference token IDs step by step.
- [ ] 18.5 Memory and latency boundaries are measured on supported hardware.
- [ ] 18.6 A clean artifact-only rehearsal preserves traceable evidence.

## Exit Criteria

The GPT-2 inference track passes when a clean installed comtam environment loads
the pinned real checkpoint through public Python APIs, matches reference
intermediates/logits/generated token IDs, rejects invalid mappings without
partial mutation, stays inside documented memory and sequence limits, and
reproduces the acceptance run from preserved artifacts and fixtures.
