import torch
from torch import nn

import ollm.dense_projection_chunking as dense_projection_chunking
from ollm.dense_projection_chunking import (
    DEFAULT_DENSE_PROJECTION_CHUNK_ROWS,
    DEFAULT_DENSE_PROJECTION_HEADROOM_RESERVE_BYTES,
    apply_gated_dense_projection,
    resolve_dense_projection_chunk_rows,
)


def test_apply_gated_dense_projection_matches_unchunked_result() -> None:
    torch.manual_seed(0)
    gate_proj = nn.Linear(4, 8, bias=False)
    up_proj = nn.Linear(4, 8, bias=False)
    down_proj = nn.Linear(8, 4, bias=False)
    activations = torch.randn(1, 5, 4, dtype=torch.float32)

    actual = apply_gated_dense_projection(
        activations,
        gate_proj=gate_proj,
        up_proj=up_proj,
        down_proj=down_proj,
        activation=torch.nn.functional.silu,
        intermediate_features=8,
        configured_chunk_rows=2,
    )
    expected = down_proj(
        torch.nn.functional.silu(gate_proj(activations.squeeze(0)))
        * up_proj(activations.squeeze(0))
    ).unsqueeze(0)

    assert torch.allclose(actual, expected)


def test_apply_gated_dense_projection_falls_back_to_full_shape_for_batch_inputs() -> (
    None
):
    torch.manual_seed(1)
    gate_proj = nn.Linear(4, 8, bias=False)
    up_proj = nn.Linear(4, 8, bias=False)
    down_proj = nn.Linear(8, 4, bias=False)
    activations = torch.randn(2, 3, 4, dtype=torch.float32)

    actual = apply_gated_dense_projection(
        activations,
        gate_proj=gate_proj,
        up_proj=up_proj,
        down_proj=down_proj,
        activation=torch.nn.functional.silu,
        intermediate_features=8,
        configured_chunk_rows=2,
    )
    expected = down_proj(
        torch.nn.functional.silu(gate_proj(activations)) * up_proj(activations)
    )

    assert torch.allclose(actual, expected)


def test_resolve_dense_projection_chunk_rows_uses_accelerator_headroom(
    monkeypatch,
) -> None:
    bytes_per_row = (2 * 2048 + 4 * 8192) * 2
    monkeypatch.setattr(
        dense_projection_chunking,
        "available_accelerator_memory_bytes",
        lambda device: (
            DEFAULT_DENSE_PROJECTION_HEADROOM_RESERVE_BYTES + (bytes_per_row * 2048)
        ),
    )

    chunk_rows = resolve_dense_projection_chunk_rows(
        total_rows=50000,
        hidden_features=2048,
        intermediate_features=8192,
        element_size=2,
        device=torch.device("cuda:0"),
        configured_chunk_rows=None,
    )

    assert chunk_rows == 2048


def test_resolve_dense_projection_chunk_rows_caps_auto_mode_at_default() -> None:
    chunk_rows = resolve_dense_projection_chunk_rows(
        total_rows=50000,
        hidden_features=2048,
        intermediate_features=8192,
        element_size=2,
        device=torch.device("cpu"),
        configured_chunk_rows=None,
    )

    assert chunk_rows == DEFAULT_DENSE_PROJECTION_CHUNK_ROWS
