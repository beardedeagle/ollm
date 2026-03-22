from ollm.runtime.benchmark.resources import (
    capture_accelerator_memory,
    reset_accelerator_metrics,
    synchronize_device,
)


def test_cuda_resource_helpers_normalize_device_index(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "torch.cuda.current_device",
        lambda: captured.setdefault("current_device", 0),
    )
    monkeypatch.setattr(
        "torch.cuda.reset_peak_memory_stats",
        lambda device: captured.setdefault("reset", device),
    )
    monkeypatch.setattr(
        "torch.cuda.synchronize",
        lambda device: captured.setdefault("sync", device),
    )
    monkeypatch.setattr("torch.cuda.memory_allocated", lambda device: 1.0)
    monkeypatch.setattr("torch.cuda.max_memory_allocated", lambda device: 2.0)
    monkeypatch.setattr("torch.cuda.memory_reserved", lambda device: 3.0)
    monkeypatch.setattr("torch.cuda.max_memory_reserved", lambda device: 4.0)

    reset_accelerator_metrics("cuda:0")
    snapshot = capture_accelerator_memory("cuda:0")
    synchronize_device("cuda:0")

    assert captured["reset"] == 0
    assert captured["sync"] == 0
    assert captured["current_device"] == 0
    assert snapshot.accelerator_kind == "cuda"
