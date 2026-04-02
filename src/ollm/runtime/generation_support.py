"""Support helpers shared by the runtime executor."""

from collections.abc import Mapping

import torch

from ollm.app.types import Message
from ollm.kv_cache.state import KVCacheStateSnapshot
from ollm.runtime.errors import PromptExecutionError

PLAN_METADATA_DETAIL_KEYS = (
    "execution_device_type",
    "specialization_device_profile",
    "strategy_selector_profile",
    "strategy_selector_rule_id",
    "strategy_selector_requested_override",
    "strategy_selector_selected_kv_cache_strategy",
    "strategy_selector_applied_kv_cache_strategy",
    "strategy_selector_fallback_chain",
    "strategy_selector_reason",
    "strategy_selector_requested_kv_cache_lifecycle",
    "strategy_selector_applied_kv_cache_lifecycle",
    "strategy_selector_lifecycle_reason",
    "strategy_selector_model_family",
    "strategy_selector_modality_bucket",
    "strategy_selector_platform",
    "strategy_selector_accelerator_kind",
    "strategy_selector_host_ram_tier",
    "strategy_selector_accelerator_memory_tier",
    "strategy_selector_required_runtime_features",
    "strategy_selector_model_size_tier",
    "offload_cpu_requested_layers",
    "offload_cpu_policy",
    "offload_cpu_total_layers",
    "offload_cpu_resolved_policy",
    "offload_cpu_applied_layers",
    "offload_cpu_applied_indices",
    "offload_gpu_layers",
)


def render_plain_prompt(messages: list[Message]) -> str:
    """Render fallback text-only prompts for tokenizers without chat templates.

    Args:
        messages (list[Message]): Prompt messages exposing `role` and
            `text_content()`.

    Returns:
        str: Plain-text prompt for non-chat-template tokenizers.
    """
    rendered_messages: list[str] = []
    for message in messages:
        text = message.text_content().strip()
        if not text:
            continue
        rendered_messages.append(f"{message.role.value.upper()}: {text}")
    rendered_messages.append("ASSISTANT:")
    return "\n\n".join(rendered_messages)


def require_tensor(value: object) -> torch.Tensor:
    """Validate that a runtime input is tensor-backed.

    Args:
        value (object): Candidate runtime input.

    Returns:
        torch.Tensor: The validated tensor value.
    """
    if not isinstance(value, torch.Tensor):
        raise PromptExecutionError("Expected tensor-backed model inputs")
    return value


def prepare_text_inputs(
    value: object, device: torch.device
) -> dict[str, torch.Tensor | object]:
    """Normalize tokenizer outputs into tensor-backed runtime inputs.

    Args:
        value (object): Tokenizer output from `apply_chat_template`.
        device (torch.device): Runtime device for tensor placement.

    Returns:
        dict[str, torch.Tensor | object]: Prepared model input mapping.
    """
    if isinstance(value, torch.Tensor):
        input_ids = value.to(device)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, device=device),
        }

    if isinstance(value, Mapping):
        prepared_inputs: dict[str, torch.Tensor | object] = {}
        for key, item in value.items():
            if isinstance(item, torch.Tensor):
                prepared_inputs[str(key)] = item.to(device)
            else:
                prepared_inputs[str(key)] = item
        if "input_ids" not in prepared_inputs:
            raise PromptExecutionError(
                "Tokenizer chat template did not return input_ids"
            )
        input_ids = require_tensor(prepared_inputs["input_ids"])
        if "attention_mask" not in prepared_inputs:
            prepared_inputs["attention_mask"] = torch.ones_like(
                input_ids, device=device
            )
        return prepared_inputs

    raise PromptExecutionError(
        "Tokenizer chat template returned unsupported model inputs"
    )


def extract_cache_state_snapshot(value: object) -> KVCacheStateSnapshot | None:
    """Extract a cache-state snapshot when the runtime cache exposes one.

    Args:
        value (object): Candidate cache object.

    Returns:
        KVCacheStateSnapshot | None: Cache-state snapshot when available.
    """
    snapshot_method = getattr(value, "cache_state_snapshot", None)
    if not callable(snapshot_method):
        return None
    snapshot = snapshot_method()
    if not isinstance(snapshot, KVCacheStateSnapshot):
        return None
    return snapshot


def normalize_generate_inputs(inputs: dict[str, object]) -> dict[str, object]:
    """Drop generation-time transport-only fields not accepted by backends.

    Args:
        inputs (dict[str, object]): Generated model input mapping.

    Returns:
        dict[str, object]: Sanitized input mapping for `generate()`.
    """
    normalized_inputs = dict(inputs)
    normalized_inputs.pop("token_type_ids", None)
    return normalized_inputs
