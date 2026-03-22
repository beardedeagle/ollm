# Weight Transport ADR

This page is contributor-facing architectural guidance.

It is not customer-facing product documentation and not an implementation
claim. It records the recommended direction for compressed or FP8-staged weight
transport in streamed native runtimes so later implementation work stays
grounded in the current oLLM architecture.

## Status

- audience: implementers and maintainers
- kind: ADR / design decision
- implementation status: not implemented

## Scope

This ADR defines the recommended direction for reducing the amount of weight
data moved during layer-streamed native execution.

It covers:

- layer-streamed weight loads from safetensors and `gds_export`
- optional transport encodings separate from execution dtype
- required safety, materialization, and benchmark changes

It does not propose a general model-format rewrite.

## Decision summary

oLLM should not start with generic "compressed transport" across all streamed
native runtimes.

The recommended direction is:

- reject generic lossless compressed transport as the first implementation slice
- treat transport encoding as separate from both authoritative on-disk model
  format and final execution dtype
- start with one explicit, manifest-backed transport contract on the custom
  `gds_export` path
- make the first concrete implementation target CUDA-only FP8-staged transport
  with explicit upcast before execution
- require transport-specific benchmark and inspection fields before any speed or
  memory claim
- require separately labeled upstream concept baselines for optimization claims

## Why the current baseline is not enough

Current streamed native runtimes still move full execution-ready weight bytes
for each loaded layer.

That limitation is visible in the current code:

- `src/ollm/llama.py` and `src/ollm/gemma3.py`
  preload and materialize one layer's safetensors, queue the next dense layer's
  reads one stage ahead when possible, assign tensors into the live module,
  then unload them again after the forward pass.
- `src/ollm/gds_loader.py`
  reads direct byte ranges from safetensor files or raw `gds_export` blobs into
  CPU or CUDA buffers.
- `src/ollm/gds_async.py`
  returns execution-ready tensors from pending read helpers; it does not expose
  a transport-decode stage yet.

The benchmark docs already identify loader-streamed safetensor IO as a real
cost on some lanes. That makes weight transport a credible future optimization
target, but only if the design stays truthful about where the bytes and latency
actually move.

## Design goals

- Reduce moved bytes on loader-streamed paths where IO or transfer is the real
  bottleneck.
- Keep transport encoding separate from execution dtype and from authoritative
  at-rest model format.
- Preserve explicit safety validation for all transport artifacts.
- Make benchmark evidence able to distinguish encoded bytes, decode work, and
  final runtime cost.
- Keep fallback behavior explicit instead of pretending an unsupported transport
  path succeeded.

## Non-goals

- Universal support across all native families in one slice.
- Replacing safetensors as the default model format.
- Hand-wavy 100B-plus claims without IO math and proof.
- Treating FP8-staged transport as end-to-end FP8 inference.
- Claiming GPU-side decompression before the repo has an explicit kernel/runtime
  path for it.

## Critical truth constraints

### Transport encoding is not at-rest format

The ADR must keep three concepts separate:

- authoritative on-disk model format
- optional streamed transport encoding
- final runtime execution dtype

If those are blurred together, the work stops being a transport design and
quietly becomes a model-format rewrite proposal.

### FP8 staging is not already supported

Today the readers map storage bytes directly into execution-ready tensors.

There is no current transport contract for:

- encoded dtype
- decode target dtype
- decode location
- scratch-buffer budget
- encoded versus decoded byte counts

So FP8-staged transport is not a loader tweak. It needs a first-class contract.

### Optimization claims require upstream baselines

Fork-internal A/B results are not enough.

Future transport work must compare:

- current fork baseline
- transport-enabled fork result
- separately labeled upstream concept baseline where applicable

The existing benchmark workflow already documents how upstream baselines stay
outside this repo tree and keep separate codebase labels.

## Credible options

| Option | Credibility in current architecture | Why |
| --- | --- | --- |
| Generic seekable lossless compression over safetensors | low | Current safetensor readers depend on raw offset-addressable bytes and would need a new seekable compressed container plus decompression-aware readers |
| FP8-staged transport with upcast before execution | medium | Credible only if modeled as an explicit transport encoding contract, not as a safetensor dtype shortcut |
| Narrow first step on `gds_export` only | high | `gds_export` already uses an explicit manifest, strict safety validation, and a dedicated GDS loader path |

## Recommended first direction

The recommended first direction is:

- define one explicit transport contract for `gds_export`
- use that contract for CUDA-only FP8-staged transport
- upcast to the current execution dtype before layer execution

Why this is the most credible first step:

- GPT-OSS already has a manifest-backed `gds_export` specialization path
- the path is already safety-gated
- the path already bypasses the generic safetensor-reader assumptions
- benchmark/reporting can extend one explicit path without pretending the whole
  runtime stack already supports transport encodings

The ADR explicitly rejects generic lossless compression as v1 because the
current CPU and GPU safetensor readers assume direct typed byte ranges, not a
compressed-block container.

## Recommended transport contract

The first transport-aware manifest should make these fields explicit per tensor:

- `encoding`
- `encoded_dtype`
- `encoded_nbytes`
- `decoded_dtype`
- `decoded_shape`
- `decoded_nbytes`
- `scale_encoding`
- `scale_path`
- `checksum`
- `decode_location`

Recommended initial `decode_location` values:

- `cuda-upcast`
- `cpu-decode-then-transfer`

The first implementation recommendation is to support only `cuda-upcast`.

That keeps the runtime claim narrow and avoids pretending CPU or MPS benefits
exist before they are proven.

## Safety and materialization implications

Current safety rules are intentionally strict:

- unsafe model artifacts are rejected
- unsafe adapter artifacts are rejected
- `gds_export` rejects torch-serialized and pickle-backed files
- `gds_export` also rejects packed `mxfp4` artifacts

Any future transport format must preserve that bar.

Required safety rules for a future transport implementation:

- transport manifests must remain inside the validated export root
- transport blobs must use explicit allowed extensions
- transport manifests must carry stable encoding identifiers and checksums
- unsafe executable, pickle-backed, or torch-serialized transport blobs remain
  forbidden
- invalid, missing, or stale sidecar transport artifacts must invalidate the
  transport path instead of being used opportunistically

Materialization implications:

- managed downloads must know how to fetch transport manifests and blobs
- completeness checks must validate both base model artifacts and transport
  sidecars
- the absence of valid transport artifacts must fall back to today's safe
  safetensor or raw `gds_export` paths

## Benchmark and inspection truth

Current benchmark surfaces already show native IO events such as:

- `gds_read`
- `safetensor_read`
- `safetensor_pread`
- `offloaded_cpu_to_cuda`

That is not enough for transport claims.

Before any transport optimization claim, the future implementation must expose:

- transport encoding used
- encoded bytes read
- decoded bytes produced
- decode or upcast time
- transport compression ratio
- scratch memory used for decode or upcast
- fallback reason when transport was requested but not used

Without those fields, oLLM would risk claiming faster or smaller transport while
hiding where the cost really moved.

## Failure and fallback behavior

The ADR makes these rules explicit:

- unsupported backend or device combinations must fall back to the current
  non-transport path
- fallback must be observable in runtime planning or benchmark output
- invalid transport manifests or blobs must reject the transport path, not be
  silently tolerated
- the runtime must never claim FP8-staged transport if it actually ran on raw
  safetensor or raw `gds_export` bytes

## Recommended phased implementation order

### Phase 1

Add transport-specific benchmark and inspection contract fields.

Recommended first slice:

- extend native runtime reporting with encoded-bytes and decode-time fields
- make fallback reasons explicit
- keep runtime behavior otherwise unchanged

### Phase 2

Define one explicit transport manifest and validation contract for `gds_export`.

Recommended first slice:

- one safe manifest schema
- one allowed transport encoding family
- one materialization completeness rule set

### Phase 3

Implement CUDA-only FP8-staged transport on GPT-OSS `gds_export`.

Recommended first slice:

- one model family
- one device class
- explicit upcast before execution
- benchmark proof against current-fork and upstream concept baselines

### Phase 4

Evaluate expansion to other streamed native families.

Do this only if:

- the new metrics show real end-to-end wins
- the quality or output delta is acceptable
- the transport contract still fits without pretending safetensors natively
  carry transport metadata

### Phase 5

Revisit lossless compression only if the repo gains one of:

- a seekable compressed-block reader contract
- an explicit GPU-side decompression path with named runtime support

Until then, generic compressed transport remains a later research option, not
the recommended first move.
