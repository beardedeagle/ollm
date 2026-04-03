import torch

from ollm.app.types import ContentPart, Message, MessageRole, PromptResponse
from ollm.runtime.backends.base import BackendRuntime
from ollm.runtime.capabilities import CapabilityProfile, SupportLevel
from ollm.runtime.generation import RuntimeExecutor
from tests.test_runtime_executor import build_request, build_runtime


def test_runtime_executor_preserves_backend_chunked_prefill_metadata() -> None:
    runtime = build_runtime(CapabilityProfile(support_level=SupportLevel.GENERIC))
    runtime.backend = BackendRuntime(
        backend_id="test-backend",
        model=None,
        tokenizer=None,
        processor=None,
        device=torch.device("cpu"),
        stats=None,
        print_suppression_modules=(),
        create_cache=lambda cache_dir, cache_strategy=None, cache_lifecycle=None, cache_window_tokens=None: (
            None
        ),
        apply_offload=lambda runtime_config: None,
        execute_prompt=lambda request, sink: PromptResponse(
            text="backend-response",
            assistant_message=Message.assistant_text("backend-response"),
            metadata={
                "chunked_prefill_strategy_id": "backend-strategy",
                "chunked_prefill_activation_reason": "backend-owned",
            },
        ),
    )
    request = build_request(
        runtime.config,
        Message(role=MessageRole.USER, content=[ContentPart.text("hello")]),
    )

    response = RuntimeExecutor().execute(runtime, request)

    assert response.metadata["chunked_prefill_strategy_id"] == "backend-strategy"
    assert response.metadata["chunked_prefill_activation_reason"] == "backend-owned"
