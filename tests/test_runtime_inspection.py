from ollm.runtime.backend_selector import BackendSelector
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.inspection import runtime_config_payload, runtime_plan_payload
from ollm.runtime.loader import RuntimeLoader
from tests.test_backend_selector import build_catalog_resolved_model


def test_runtime_config_payload_includes_cpu_offload_policy() -> None:
    payload = runtime_config_payload(
        RuntimeConfig(
            device="mps",
            offload_cpu_policy="suffix",
            dense_projection_chunk_rows=2048,
        )
    )

    assert payload["offload_cpu_policy"] == "suffix"
    assert payload["dense_projection_chunk_rows"] == 2048


def test_runtime_config_payload_marks_auto_strategy_when_unset() -> None:
    payload = runtime_config_payload(RuntimeConfig())

    assert payload["kv_cache_strategy"] == "auto"
    assert payload["strategy_selector_profile"] == "balanced"


def test_runtime_plan_payload_includes_details() -> None:
    runtime_plan = BackendSelector().select(
        build_catalog_resolved_model(),
        RuntimeConfig(device="mps", offload_cpu_layers=2, offload_cpu_policy="suffix"),
    )

    payload = runtime_plan_payload(runtime_plan)

    assert payload["details"]["offload_cpu_policy"] == "suffix"


def test_runtime_loader_plan_includes_selector_details(tmp_path) -> None:
    runtime_plan = RuntimeLoader().plan(
        RuntimeConfig(
            model_reference="llama3-1B-chat",
            models_dir=tmp_path / "models",
            device="cpu",
        )
    )

    payload = runtime_plan_payload(runtime_plan)

    assert payload["details"]["strategy_selector_profile"] == "balanced"
    assert "strategy_selector_rule_id" in payload["details"]
