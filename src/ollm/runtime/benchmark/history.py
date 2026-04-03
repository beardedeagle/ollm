"""Persistent benchmark history and regression comparison helpers."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ollm.async_io import path_append_text, path_mkdir, path_write_text
from ollm.runtime.benchmark.history_summary import (
    compare_metric_summaries,
    require_object,
    summarize_benchmark_payload,
)
from ollm.runtime.benchmark.metadata import (
    build_git_summary,
    build_history_codebase_summary,
    build_history_host_summary,
)

_HISTORY_DIR = Path(".ollm/benchmark-history")
_LATEST_DIRNAME = "latest"


def resolve_benchmark_history_dir(
    *,
    cli_history_dir: Path | None,
    configured_history_dir: Path | None,
) -> Path | None:
    if cli_history_dir is not None:
        return cli_history_dir
    return configured_history_dir


def record_benchmark_history(
    *,
    repo_root: Path,
    payload: dict[str, object],
    run_kind: str,
    history_dir: Path | None,
    comparison_key: dict[str, object],
    codebase_label: str,
) -> dict[str, object]:
    """Persist one benchmark payload and compare it with the last matching run."""

    resolved_history_dir = (
        (repo_root / _HISTORY_DIR).resolve()
        if history_dir is None
        else history_dir.expanduser().resolve()
    )
    records_dir = resolved_history_dir / "records"
    latest_dir = resolved_history_dir / _LATEST_DIRNAME
    index_path = resolved_history_dir / "index.jsonl"
    path_mkdir(records_dir, parents=True, exist_ok=True)
    path_mkdir(latest_dir, parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    previous = find_previous_record(index_path, comparison_key=comparison_key)
    summary = summarize_benchmark_payload(payload, run_kind=run_kind)
    comparison = (
        None
        if previous is None
        else compare_metric_summaries(
            current=summary,
            previous=require_object(previous.get("summary"), "summary"),
        )
    )
    record = {
        "generated_at": generated_at,
        "run_kind": run_kind,
        "comparison_key": comparison_key,
        "codebase": build_history_codebase_summary(
            repo_root, codebase_label=codebase_label
        ),
        "host": build_history_host_summary(),
        "git": build_git_summary(repo_root),
        "summary": summary,
        "comparison_to_previous": comparison,
        "payload": payload,
    }
    record_name = (
        f"{generated_at.replace(':', '').replace('+00:00', 'Z')}-{run_kind}.json"
    )
    record_path = records_dir / record_name
    rendered_record = json.dumps(record, indent=2, sort_keys=True) + "\n"
    path_write_text(record_path, rendered_record, encoding="utf-8")
    index_entry: dict[str, object] = {
        "generated_at": generated_at,
        "record_path": str(record_path),
        "run_kind": run_kind,
        "comparison_key": comparison_key,
        "codebase": record["codebase"],
        "summary": summary,
        "comparison_to_previous": comparison,
        "host": record["host"],
        "git": record["git"],
    }
    _append_jsonl_entry(index_path, index_entry)
    _write_latest_entry(index_path, comparison_key=comparison_key, entry=index_entry)
    return {
        "record_path": str(record_path),
        "codebase_label": codebase_label,
        "comparison_to_previous": comparison,
        "summary": summary,
    }


def find_previous_record(
    index_path: Path, *, comparison_key: dict[str, object]
) -> dict[str, object] | None:
    """Return the last matching benchmark history entry, if any."""

    latest_entry = _read_latest_entry(index_path, comparison_key=comparison_key)
    if latest_entry is not None:
        return latest_entry
    if not index_path.exists():
        return None
    for line in reversed(index_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        entry = json.loads(line)
        if not isinstance(entry, dict):
            continue
        if entry.get("comparison_key") == comparison_key:
            return entry
    return None


def _append_jsonl_entry(path: Path, payload: dict[str, object]) -> None:
    rendered_payload = json.dumps(payload, sort_keys=True)
    line = rendered_payload + "\n"
    if not path.exists():
        path_write_text(path, line, encoding="utf-8")
        return
    path_append_text(path, line, encoding="utf-8")


def _comparison_key_digest(comparison_key: dict[str, object]) -> str:
    rendered = json.dumps(comparison_key, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _latest_entry_path(index_path: Path, *, comparison_key: dict[str, object]) -> Path:
    return (
        index_path.parent
        / _LATEST_DIRNAME
        / f"{_comparison_key_digest(comparison_key)}.json"
    )


def _read_latest_entry(
    index_path: Path, *, comparison_key: dict[str, object]
) -> dict[str, object] | None:
    latest_path = _latest_entry_path(index_path, comparison_key=comparison_key)
    if not latest_path.exists():
        return None
    try:
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("comparison_key") != comparison_key:
        return None
    return payload


def _write_latest_entry(
    index_path: Path,
    *,
    comparison_key: dict[str, object],
    entry: dict[str, object],
) -> None:
    latest_path = _latest_entry_path(index_path, comparison_key=comparison_key)
    rendered_entry = json.dumps(entry, indent=2, sort_keys=True) + "\n"
    path_write_text(latest_path, rendered_entry, encoding="utf-8")
