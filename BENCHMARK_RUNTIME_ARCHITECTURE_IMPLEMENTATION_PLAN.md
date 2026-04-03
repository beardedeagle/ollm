# Benchmark Runtime Architecture Implementation Plan

## Purpose

This document captures the implementation plan for addressing four structural issues in the runtime benchmark subsystem centered on [scripts/benchmark_runtime.py](scripts/benchmark_runtime.py) and `src/ollm/runtime/benchmark/`.

The goal is to preserve current benchmark behavior and report contracts while improving architectural boundaries, maintainability, and scaling characteristics.

## Scope

This plan addresses these four findings:

1. Benchmark probe execution is tightly coupled to private `RuntimeExecutor` internals.
2. Probe-mode handling is stringly typed and duplicated across CLI, history extraction, and command construction.
3. `src/ollm/runtime/benchmark/__init__.py` is overloaded and imports heavyweight fixture code at package import time.
4. Benchmark history lookup scales poorly because it reverse-scans the full `index.jsonl` on every comparison lookup.

This plan does not change benchmark semantics, output meaning, or supported benchmark scenarios. It is a structural refactor with behavior preservation.

## Current Problems

### 1. Private Runtime Coupling

The benchmark layer currently calls private executor methods in [src/ollm/runtime/benchmark/probe_execution.py](src/ollm/runtime/benchmark/probe_execution.py), including request validation, input construction, generation kwargs preparation, and response decoding.

Problems:

- Internal runtime refactors can silently break benchmark code.
- The benchmark layer knows too much about execution internals.
- There is no stable public runtime seam for traced single-request execution.

### 2. Probe Mode Duplication

Probe mode strings such as `cold`, `warm`, `prompt-scaling`, `output-scaling`, `session-growth`, and `reopen-session-growth` are manually coordinated across:

- [scripts/benchmark_runtime.py](scripts/benchmark_runtime.py)
- [scripts/benchmark_runtime_support.py](scripts/benchmark_runtime_support.py)
- [src/ollm/runtime/benchmark/targets.py](src/ollm/runtime/benchmark/targets.py)
- the probe parse/render paths under `src/ollm/runtime/benchmark/`

Problems:

- Adding a new probe mode requires editing multiple dispatch sites.
- String mismatches are easy to introduce.
- The mode-specific contract is not centralized.

### 3. Overloaded Package Boundary

[src/ollm/runtime/benchmark/__init__.py](src/ollm/runtime/benchmark/__init__.py) currently does too much:

- report orchestration
- host summary
- tiny fixture generation
- public re-exports
- heavy imports from `transformers` and `tokenizers`

Problems:

- Import cost is higher than needed.
- Package boundaries are unclear.
- Test fixtures and orchestration logic are mixed into the package entrypoint.

### 4. History Lookup Scalability

[src/ollm/runtime/benchmark/history.py](src/ollm/runtime/benchmark/history.py) currently finds the previous comparable record by reverse-reading the full `index.jsonl`.

Problems:

- Lookup work grows linearly with benchmark history size.
- The append-only audit trail is fine, but lookup performance will degrade as history accumulates.

## Design Goals

- Preserve existing benchmark behavior and JSON contracts.
- Introduce clear public seams between runtime execution and benchmark measurement.
- Expose only typed, benchmark-relevant execution facts, not dict-shaped runtime internals.
- Centralize probe mode metadata and dispatch.
- Make package import behavior lighter and more intentional.
- Keep the existing append-only benchmark history trail while making steady-state lookup fast.
- Avoid new dependencies.

## Target Architecture

### A. Public Runtime Execution Trace Seam

Introduce a new runtime-level public API for single-request traced execution.

Proposed module:

- [src/ollm/runtime/execution_trace.py](src/ollm/runtime/execution_trace.py)

Proposed responsibilities:

- validate a `PromptRequest`
- build normalized inputs
- build generation kwargs
- execute generation
- return only the stable execution facts needed by the benchmark layer

Proposed public shape:

```python
@dataclass(frozen=True, slots=True)
class RuntimeExecutionTrace:
    generation_started_at: float
    prompt_token_count: int
    decode_prefix_token_count: int
    output_token_count: int
    response_text: str
    cache_state: KVCacheStateSnapshot | None


def execute_request_with_trace(
    *,
    runtime: LoadedRuntime,
    request: PromptRequest,
    streamer: BufferedTextStreamer | None = None,
) -> RuntimeExecutionTrace:
    ...
```

Field intent:

- `prompt_token_count`: prompt tokens before any chunked-prefill reduction, used for prompt-throughput metrics
- `decode_prefix_token_count`: the effective `input_ids` length used for decode/output accounting after any runtime-side preparation
- `output_token_count`: generated token count after model-kind-specific accounting, so benchmark code does not inspect raw model outputs
- `response_text`: fully decoded response text using the runtime's actual decode path
- `cache_state`: extracted `KVCacheStateSnapshot` from the effective cache object when available

The seam may still internally build and normalize model inputs and generation kwargs, but those stay private to the runtime layer. The benchmark layer should not receive `prepared_inputs`, `prepared_generate_kwargs`, or raw output tensors unless a concrete benchmark requirement emerges that cannot be expressed as a smaller typed field.

Benchmark code in [src/ollm/runtime/benchmark/probe_execution.py](src/ollm/runtime/benchmark/probe_execution.py) should consume only this public seam.

### B. Central Probe Registry

Introduce a typed registry for probe modes.

Proposed module:

- [src/ollm/runtime/benchmark/probe_registry.py](src/ollm/runtime/benchmark/probe_registry.py)

Proposed structures:

```python
class ProbeMode(str, Enum):
    COLD = "cold"
    WARM = "warm"
    PROMPT_SCALING = "prompt-scaling"
    OUTPUT_SCALING = "output-scaling"
    SESSION_GROWTH = "session-growth"
    REOPEN_SESSION_GROWTH = "reopen-session-growth"


@dataclass(frozen=True, slots=True)
class ProbeDefinition:
    mode: ProbeMode
    run_kind: str
    runner: Callable[..., object]
    renderer: Callable[[object], str]
    history_request_extractor: Callable[[Mapping[str, object]], Mapping[str, object] | None]
```

Registry responsibilities:

- canonical CLI value
- probe runner
- probe renderer
- history request extraction
- any mode-specific argument interpretation

This removes repeated string dispatch from CLI, support helpers, and target command construction.

### C. Thin Benchmark Package Boundary

Split orchestration code out of `__init__.py`.

Proposed modules:

- [src/ollm/runtime/benchmark/report_builder.py](src/ollm/runtime/benchmark/report_builder.py)
- [src/ollm/runtime/benchmark/host.py](src/ollm/runtime/benchmark/host.py)
- [src/ollm/runtime/benchmark/fixtures.py](src/ollm/runtime/benchmark/fixtures.py)

Target responsibility split:

- `report_builder.py`: build full runtime benchmark report
- `host.py`: host/device summary helpers
- `fixtures.py`: tiny T5 fixture generation and any heavy fixture-only imports
- `__init__.py`: re-exports only

Heavy `transformers` and `tokenizers` imports should live in fixture-oriented code paths, and preferably be localized inside the fixture creation path where practical.

### D. Constant-Time Latest History Lookup

Retain:

- append-only records under `records/`
- append-only `index.jsonl`

Add:

- `latest/` sidecar entries keyed by stable hash of the `comparison_key`

Proposed behavior:

- on write:
  - append full entry to `index.jsonl`
  - write/update `latest/<comparison-key-hash>.json`
- on read:
  - check `latest/<comparison-key-hash>.json` first
  - fall back to `index.jsonl` reverse scan only for compatibility or repair

This keeps the audit trail intact while making normal previous-record lookup effectively constant time.

## Implementation Phases

### Phase 0. Lock Current Behavior

Add or strengthen characterization tests before structural edits.

Required coverage:

- CLI invalid argument handling in [tests/test_benchmark_reporting.py](tests/test_benchmark_reporting.py)
- JSON render/parse round-trips for all probe types
- history comparison behavior in [tests/test_benchmark_history.py](tests/test_benchmark_history.py)
- a regression test asserting benchmark code does not depend on `RuntimeExecutor._*` after the seam refactor

Deliverable:

- current observable behavior is pinned by tests before refactor work begins

### Phase 1. Thin Package Boundary

Move orchestration and fixtures out of `__init__.py`.

Changes:

- move report-building logic to `report_builder.py`
- move host summary helpers to `host.py`
- move tiny fixture generation to `fixtures.py`
- reduce `__init__.py` to deliberate re-exports

Acceptance criteria:

- importing `ollm.runtime.benchmark` no longer pulls in fixture implementation code directly
- external imports continue to work through re-exports

### Phase 2. History Lookup Improvement

Add the sidecar latest-record index.

Changes:

- add stable comparison-key hashing helper
- add `latest/` directory support in history recording
- read latest sidecar first in `find_previous_record()`
- retain `index.jsonl` append-only history

Acceptance criteria:

- normal previous-record lookup does not read all historical lines
- older history remains readable
- benchmark history tests cover the sidecar path and fallback path

### Phase 3. Probe Registry

Centralize probe-mode metadata and dispatch.

Changes:

- add `ProbeMode` enum
- add registry of `ProbeDefinition` entries
- replace raw string dispatch in [scripts/benchmark_runtime.py](scripts/benchmark_runtime.py)
- replace `_probe_request_payload()` switch logic in [scripts/benchmark_runtime_support.py](scripts/benchmark_runtime_support.py)
- update `build_runtime_probe_command()` in [src/ollm/runtime/benchmark/targets.py](src/ollm/runtime/benchmark/targets.py) to use typed modes

Acceptance criteria:

- one registry entry defines each probe mode contract
- CLI dispatch, history extraction, and command construction are registry-driven
- adding a future probe mode requires one implementation and one registry entry, not several separate switches

### Phase 4. Public Runtime Execution Seam

Replace private benchmark coupling with a stable public runtime trace API.

Changes:

- add `execution_trace.py` or equivalent runtime seam
- move execution internals needed by benchmarks behind public runtime helpers
- update [src/ollm/runtime/benchmark/probe_execution.py](src/ollm/runtime/benchmark/probe_execution.py) to use the seam
- remove benchmark references to `RuntimeExecutor._validate_request`, `_build_inputs`, `_build_generate_kwargs`, and `_decode_response`

Acceptance criteria:

- benchmark code uses only public runtime interfaces
- runtime internals remain encapsulated
- behavior of benchmark metrics remains unchanged

## File-Level Change Map

### New Files

- [src/ollm/runtime/execution_trace.py](src/ollm/runtime/execution_trace.py)
- [src/ollm/runtime/benchmark/probe_registry.py](src/ollm/runtime/benchmark/probe_registry.py)
- [src/ollm/runtime/benchmark/report_builder.py](src/ollm/runtime/benchmark/report_builder.py)
- [src/ollm/runtime/benchmark/host.py](src/ollm/runtime/benchmark/host.py)
- [src/ollm/runtime/benchmark/fixtures.py](src/ollm/runtime/benchmark/fixtures.py)

### Primary Modified Files

- [scripts/benchmark_runtime.py](scripts/benchmark_runtime.py)
- [scripts/benchmark_runtime_support.py](scripts/benchmark_runtime_support.py)
- [src/ollm/runtime/benchmark/__init__.py](src/ollm/runtime/benchmark/__init__.py)
- [src/ollm/runtime/benchmark/history.py](src/ollm/runtime/benchmark/history.py)
- [src/ollm/runtime/benchmark/probe_execution.py](src/ollm/runtime/benchmark/probe_execution.py)
- [src/ollm/runtime/benchmark/targets.py](src/ollm/runtime/benchmark/targets.py)

### Expected Test Updates

- [tests/test_benchmarks.py](tests/test_benchmarks.py)
- [tests/test_benchmark_reporting.py](tests/test_benchmark_reporting.py)
- [tests/test_benchmark_history.py](tests/test_benchmark_history.py)

## Risks

### Highest Risk

Replacing benchmark dependence on `RuntimeExecutor` private methods.

Risk:

- subtle changes in prompt token counting, decoding, streamer timing, or cache-state extraction

Mitigation:

- land characterization coverage first
- keep the new runtime seam narrow
- move logic behind a public runtime seam rather than re-implementing it in the benchmark package

### Medium Risk

Probe registry refactor.

Risk:

- mode-specific history extraction or rendering could drift

Mitigation:

- keep registry entries explicit and test every mode end-to-end

### Low Risk

`__init__.py` split and history sidecar index.

Risk:

- import path churn
- sidecar/index divergence if write ordering is wrong

Mitigation:

- preserve re-export compatibility
- write record, append index, update sidecar in a consistent order
- test fallback behavior

## Non-Goals

- no benchmark metric definition changes
- no new benchmark scenarios
- no docs rewrite beyond parity updates required by file/module moves
- no new external dependencies

## Verification Plan

Minimum full gate after the refactor:

```bash
uv run python scripts/check_python_standards.py
uv run ruff check src tests examples scripts
uv run ty check src tests scripts
uv run python -m compileall src tests scripts
uv run pytest -q
uv build
uv run python -m pip_audit
uv run --group docs mkdocs build --strict
git diff --check
```

Additional targeted checks:

```bash
uv run pytest -q tests/test_benchmarks.py tests/test_benchmark_reporting.py tests/test_benchmark_history.py tests/test_benchmark_resources.py
```

Acceptance verification:

- benchmark CLI behavior is unchanged from the user perspective
- benchmark JSON shapes remain stable
- benchmark history comparison still works
- benchmark code no longer uses `RuntimeExecutor._*`
- import of `ollm.runtime.benchmark` is slimmer and more intentional

## Recommended PR Strategy

Preferred split:

1. `benchmark: split package boundary and fixture code`
2. `benchmark: add constant-time latest history lookup`
3. `benchmark: centralize probe mode registry`
4. `runtime: add public execution trace seam for benchmarks`

Alternative:

- one larger branch is acceptable if the behavior-locking tests land first and the final diff remains reviewable

## Estimated Implementation Effort

If implemented directly in this repo with full verification on every push:

- public runtime execution seam: 5 to 8 hours
- probe registry refactor: 2 to 4 hours
- `__init__.py` split: 1 to 2 hours
- history sidecar lookup: 1 to 2 hours
- tests, verification, and cleanup fallout: 1 to 2 hours

Expected total:

- approximately 10 to 16 hours
- realistically 1.5 to 2.5 working days

## Definition of Done

This work is complete when:

- all four structural issues are resolved
- benchmark behavior is unchanged except for internal architecture improvement
- benchmark code has a stable public runtime seam
- probe-mode logic is centralized
- benchmark package imports are properly layered
- benchmark history lookup is fast in the steady state
- the full verification stack passes
