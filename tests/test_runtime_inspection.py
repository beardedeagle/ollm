from ollm.runtime.backend_selector import BackendSelector
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.inspection import runtime_config_payload, runtime_plan_payload
from tests.test_backend_selector import build_catalog_resolved_model


def test_runtime_config_payload_includes_cpu_offload_policy() -> None:
    payload = runtime_config_payload(
        RuntimeConfig(device="mps", offload_cpu_policy="suffix")
    )

    assert payload["offload_cpu_policy"] == "suffix"


def test_runtime_plan_payload_includes_details() -> None:
    runtime_plan = BackendSelector().select(
        build_catalog_resolved_model(),
        RuntimeConfig(device="mps", offload_cpu_layers=2, offload_cpu_policy="suffix"),
    )

    payload = runtime_plan_payload(runtime_plan)

    assert payload["details"]["offload_cpu_policy"] == "suffix"
