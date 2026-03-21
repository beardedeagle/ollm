"""Offload-specific benchmark summary helpers."""

from ollm.runtime.benchmark_probe_types import RequestProbeMetrics


def summarize_request_offload(samples: list[RequestProbeMetrics]) -> dict[str, object]:
    from ollm.runtime.benchmark_details import (
        optional_summary_dict,
        single_optional_string,
    )

    return {
        "cpu_policy": single_optional_string(
            [
                sample.offload_cpu_policy
                for sample in samples
                if sample.offload_cpu_policy is not None
            ]
        ),
        "cpu_requested_layers": optional_summary_dict(
            [
                float(sample.offload_cpu_requested_layers)
                for sample in samples
                if sample.offload_cpu_requested_layers is not None
            ]
        ),
        "cpu_applied_layers": optional_summary_dict(
            [
                float(sample.offload_cpu_applied_layers)
                for sample in samples
                if sample.offload_cpu_applied_layers is not None
            ]
        ),
        "cpu_applied_indices": single_optional_string(
            [
                ",".join(
                    str(layer_idx) for layer_idx in sample.offload_cpu_applied_indices
                )
                for sample in samples
                if sample.offload_cpu_applied_indices
            ]
        ),
    }
