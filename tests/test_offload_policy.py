from ollm.runtime.offload_policy import (
    format_layer_indices,
    normalize_cpu_offload_policy,
    plan_cpu_offload_placement,
)


def test_plan_cpu_offload_placement_prefix_policy() -> None:
    placement = plan_cpu_offload_placement(
        requested_layers=3,
        total_layers=8,
        policy="prefix",
    )

    assert placement.resolved_policy_id == "prefix"
    assert placement.layer_indices == (0, 1, 2)


def test_plan_cpu_offload_placement_suffix_policy() -> None:
    placement = plan_cpu_offload_placement(
        requested_layers=2,
        total_layers=8,
        policy="suffix",
    )

    assert placement.resolved_policy_id == "suffix"
    assert placement.layer_indices == (6, 7)


def test_plan_cpu_offload_placement_auto_resolves_to_middle_band() -> None:
    placement = plan_cpu_offload_placement(
        requested_layers=2,
        total_layers=8,
        policy="auto",
    )

    assert placement.requested_policy_id == "auto"
    assert placement.resolved_policy_id == "middle-band"
    assert placement.layer_indices == (3, 4)


def test_normalize_cpu_offload_policy_rejects_unknown_value() -> None:
    try:
        normalize_cpu_offload_policy("unknown")
    except ValueError as exc:
        assert "--offload-cpu-policy must be one of" in str(exc)
    else:
        raise AssertionError("Expected an unknown CPU offload policy to fail")


def test_format_layer_indices_serializes_indices() -> None:
    assert format_layer_indices((1, 3, 5)) == "1,3,5"
