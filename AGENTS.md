# comtam

Read this file and `README.md` before working in this repository.

## Project

`comtam` is an experimental eager-mode tensor framework in C++20 for Apple
Silicon. Treat it as contributor-facing research software with an unstable API
and an explicit non-production scope.

Preserve these constraints unless the request explicitly changes them:

- C++20;
- Apple GPU through `metal-cpp`;
- float32 first;
- one device first;
- shared Metal buffers first;
- eager runtime, not a lazy graph or compiler;
- correctness before optimization.

## Engineering priorities

- Keep storage, view, tensor identity, runtime, and ownership boundaries
  explicit.
- Prefer a direct implementation until repetition or evidence earns an
  abstraction.
- Give every Metal object one obvious owner.
- Reject mixed-runtime operations before allocation or dispatch.
- Add a correctness test for every new operation.
- Add an independent gradient test for every future autograd rule.
- Do not make performance claims without a reproducible benchmark.

Avoid speculative Python bindings, additional backends, dynamic backend
loading, broad dtype support, serialization, allocator caches, and production
infrastructure.

## Contributions

Inspect the current source, tests, and design notes before changing an
architectural contract. Existing code is not self-justifying: a divergence is
accepted only with a concrete mechanism, tradeoff, and relevant test evidence.

Keep changes focused. Do not mix semantic changes with unrelated formatting or
cleanup. Treat review requests as read-only unless edits are explicitly
requested, and preserve unrelated work in dirty worktrees.

## Verification

For code changes:

```sh
./build.sh
./test.sh
```

Report which commands ran. Distinguish compilation, smoke execution, and test
success. Printed arrays are diagnostics, not tests. Prefer MLX or another
independent oracle with exact or tolerance-based comparisons.

If Metal execution is unavailable, state what could and could not be verified;
do not describe an unexecuted path as tested.

## Project skills

Reusable project procedures live under `.agents/skills/`. When a task touches
Metal or metal-cpp host code, read
`.agents/skills/comtam-metal/SKILL.md` before editing. Add future skills only
for repeatable, project-specific work whose guidance changes engineering
decisions; keep product contracts in source, tests, and contributor docs.

## Style

- Match the existing C++ and Metal style.
- Keep comments sparse and mechanism-focused.
- Use `rg` for search.
- Use `apply_patch` for manual edits.
- Never revert user changes unless explicitly asked.
