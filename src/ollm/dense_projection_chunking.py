"""Helpers for adaptive row chunking in large gated dense projections."""

from collections.abc import Callable

import torch

from ollm.kv_cache.policy import available_accelerator_memory_bytes

DEFAULT_DENSE_PROJECTION_CHUNK_ROWS = 16384
DEFAULT_DENSE_PROJECTION_HEADROOM_RESERVE_BYTES = 512 * 1024 * 1024


def normalize_dense_projection_chunk_rows(rows: int | None) -> int | None:
    """Validate an optional explicit dense-projection row budget.

    Args:
        rows: Optional explicit maximum number of activation rows to process per
            chunk.

    Returns:
        int | None: The validated row budget, or ``None`` when the runtime
            should derive the chunk budget automatically.

    Raises:
        ValueError: Raised when ``rows`` is zero or negative.
    """
    if rows is None:
        return None
    if rows <= 0:
        raise ValueError("dense_projection_chunk_rows must be greater than zero")
    return rows


def resolve_dense_projection_chunk_rows(
    *,
    total_rows: int,
    hidden_features: int,
    intermediate_features: int,
    element_size: int,
    device: torch.device,
    configured_chunk_rows: int | None,
) -> int:
    """Resolve the row budget for a chunked gated dense projection.

    Args:
        total_rows: Total activation rows awaiting projection.
        hidden_features: Input or output hidden width for the projection stack.
        intermediate_features: Expanded gated MLP width.
        element_size: Tensor element size in bytes for the active dtype.
        device: Execution device for the current activations.
        configured_chunk_rows: Optional explicit chunk-row override.

    Returns:
        int: The number of rows to process per chunk. The returned value is
            always at least one and never exceeds ``total_rows``.
    """
    if total_rows <= 0:
        raise ValueError("total_rows must be greater than zero")
    normalized_override = normalize_dense_projection_chunk_rows(configured_chunk_rows)
    if normalized_override is not None:
        return min(total_rows, normalized_override)

    default_rows = min(total_rows, DEFAULT_DENSE_PROJECTION_CHUNK_ROWS)
    available_bytes = available_accelerator_memory_bytes(device)
    if available_bytes is None:
        return default_rows
    bytes_per_row = _estimate_gated_projection_bytes_per_row(
        hidden_features=hidden_features,
        intermediate_features=intermediate_features,
        element_size=element_size,
    )
    reserved_bytes = max(
        DEFAULT_DENSE_PROJECTION_HEADROOM_RESERVE_BYTES,
        available_bytes // 8,
    )
    usable_bytes = available_bytes - reserved_bytes
    if usable_bytes <= 0:
        return 1
    adaptive_rows = usable_bytes // bytes_per_row
    if adaptive_rows <= 0:
        return 1
    return max(1, min(default_rows, int(adaptive_rows)))


def apply_gated_dense_projection(
    x: torch.Tensor,
    *,
    gate_proj: Callable[[torch.Tensor], torch.Tensor],
    up_proj: Callable[[torch.Tensor], torch.Tensor],
    down_proj: Callable[[torch.Tensor], torch.Tensor],
    activation: Callable[[torch.Tensor], torch.Tensor],
    intermediate_features: int,
    configured_chunk_rows: int | None,
) -> torch.Tensor:
    """Apply a gated dense projection with adaptive row chunking.

    Args:
        x: Input activations. The optimized-native dense runtimes currently emit
            either `[1, tokens, hidden]` or `[tokens, hidden]` tensors here.
        gate_proj: Gate projection callable.
        up_proj: Up projection callable.
        down_proj: Down projection callable.
        activation: Activation function used after the gate projection.
        intermediate_features: Expanded width for the gated MLP stack.
        configured_chunk_rows: Optional explicit maximum rows per chunk.

    Returns:
        torch.Tensor: Projected activations with the same rank shape contract as
            the incoming tensor.
    """
    prepared = _prepare_chunkable_activations(x)
    if prepared is None:
        return down_proj(activation(gate_proj(x)) * up_proj(x))
    activations, restore_batch_axis = prepared
    chunk_rows = resolve_dense_projection_chunk_rows(
        total_rows=activations.shape[0],
        hidden_features=activations.shape[-1],
        intermediate_features=intermediate_features,
        element_size=activations.element_size(),
        device=activations.device,
        configured_chunk_rows=configured_chunk_rows,
    )
    result = (
        _project_dense_chunk(
            activations,
            gate_proj=gate_proj,
            up_proj=up_proj,
            down_proj=down_proj,
            activation=activation,
        )
        if chunk_rows >= activations.shape[0]
        else _project_dense_chunks(
            activations,
            gate_proj=gate_proj,
            up_proj=up_proj,
            down_proj=down_proj,
            activation=activation,
            chunk_rows=chunk_rows,
        )
    )
    if restore_batch_axis:
        return result.unsqueeze(0)
    return result


def _estimate_gated_projection_bytes_per_row(
    *,
    hidden_features: int,
    intermediate_features: int,
    element_size: int,
) -> int:
    if hidden_features <= 0 or intermediate_features <= 0 or element_size <= 0:
        return 1
    # Input activations plus gate/up/product intermediates and the down-projected
    # output. Keep one extra intermediate-sized buffer in the estimate to bias
    # the adaptive path toward the safer side when memory is tight.
    estimated_elements = (2 * hidden_features) + (4 * intermediate_features)
    return max(1, estimated_elements * element_size)


def _prepare_chunkable_activations(
    x: torch.Tensor,
) -> tuple[torch.Tensor, bool] | None:
    if x.ndim == 2:
        return x, False
    if x.ndim == 3 and x.shape[0] == 1:
        return x.squeeze(0), True
    return None


def _project_dense_chunk(
    activations: torch.Tensor,
    *,
    gate_proj: Callable[[torch.Tensor], torch.Tensor],
    up_proj: Callable[[torch.Tensor], torch.Tensor],
    down_proj: Callable[[torch.Tensor], torch.Tensor],
    activation: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    gate_chunk = activation(gate_proj(activations))
    up_chunk = up_proj(activations)
    return down_proj(gate_chunk * up_chunk)


def _project_dense_chunks(
    activations: torch.Tensor,
    *,
    gate_proj: Callable[[torch.Tensor], torch.Tensor],
    up_proj: Callable[[torch.Tensor], torch.Tensor],
    down_proj: Callable[[torch.Tensor], torch.Tensor],
    activation: Callable[[torch.Tensor], torch.Tensor],
    chunk_rows: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for index in range(0, activations.shape[0], chunk_rows):
        chunk = activations[index : index + chunk_rows]
        chunks.append(
            _project_dense_chunk(
                chunk,
                gate_proj=gate_proj,
                up_proj=up_proj,
                down_proj=down_proj,
                activation=activation,
            )
        )
    return torch.cat(chunks, dim=0)
