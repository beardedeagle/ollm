from dataclasses import replace

from ollm.app.types import ContentPart, Message, MessageRole
from ollm.runtime.capabilities import CapabilityProfile, SupportLevel
from ollm.runtime.generation import RuntimeExecutor
from tests.test_runtime_executor import build_request, build_runtime
from tests.test_runtime_executor_metadata import FakeCache


def test_runtime_executor_reuses_disk_cache_within_loaded_runtime() -> None:
    runtime = build_runtime(CapabilityProfile(support_level=SupportLevel.OPTIMIZED))
    runtime.config.use_cache = True
    runtime.config.kv_cache_strategy = "log-structured-journal"
    runtime.plan = replace(runtime.plan, supports_disk_cache=True)
    create_cache_calls: list[tuple[object, object, object]] = []

    def _create_cache(cache_dir, cache_strategy=None, cache_lifecycle=None):
        create_cache_calls.append((cache_dir, cache_strategy, cache_lifecycle))
        return FakeCache()

    runtime.backend.create_cache = _create_cache
    request = build_request(
        runtime.config,
        Message(role=MessageRole.USER, content=[ContentPart.text("hello")]),
    )

    first = RuntimeExecutor().execute(runtime, request)
    second = RuntimeExecutor().execute(runtime, request)

    assert first.metadata["kv_cache_strategy"] == "log-structured-journal"
    assert second.metadata["kv_cache_strategy"] == "log-structured-journal"
    assert len(create_cache_calls) == 1
    assert create_cache_calls[0][2] == "runtime-scoped"
