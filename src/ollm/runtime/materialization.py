"""Managed Hugging Face model materialization helpers."""

import json
import logging
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

from huggingface_hub import snapshot_download
from transformers import AutoConfig

LOGGER = logging.getLogger(__name__)

HF_RUNTIME_ARTIFACT_PATTERNS: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "*.safetensors",
    "*.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "vocab.txt",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "processor_config.json",
    "preprocessor_config.json",
    "feature_extractor.json",
    "feature_extractor_config.json",
    "image_processor_config.json",
    "chat_template.json",
    "chat_template.jinja",
    "gds_export",
    "gds_export/*",
    "gds_export/**",
    "gds_export/**/*",
)
HF_TOKENIZER_RUNTIME_FILES: tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "vocab.txt",
)
HF_TOKENIZER_PAIR_FILES: tuple[str, ...] = ("vocab.json", "merges.txt")
HF_PROCESSOR_RUNTIME_FILES: tuple[str, ...] = (
    "processor_config.json",
    "preprocessor_config.json",
    "feature_extractor.json",
    "feature_extractor_config.json",
    "image_processor_config.json",
)


class _SnapshotDownloadCallable(Protocol):
    def __call__(
        self,
        *,
        repo_id: str,
        local_dir: str,
        local_dir_use_symlinks: bool,
        force_download: bool,
        revision: str | None,
        allow_patterns: list[str],
    ) -> str: ...


class ManagedModelDownloadError(RuntimeError):
    """Raised when a managed Hugging Face materialization is incomplete."""

    def __init__(
        self,
        repo_id: str,
        model_dir: Path,
        artifact_gaps: tuple[str, ...],
    ) -> None:
        self.repo_id = repo_id
        self.model_dir = model_dir
        self.artifact_gaps = artifact_gaps
        gap_summary = "; ".join(artifact_gaps)
        super().__init__(
            f"Managed model download for '{repo_id}' at {model_dir} is incomplete: "
            f"{gap_summary}. If this is a gated Hugging Face repository, ensure the "
            "terms are accepted and the active Hugging Face token has access to the model weights."
        )


def _is_hf_runtime_artifact(relative_path: PurePosixPath) -> bool:
    return any(relative_path.match(pattern) for pattern in HF_RUNTIME_ARTIFACT_PATTERNS)


def prune_hf_runtime_artifacts(model_dir: str | Path) -> tuple[Path, ...]:
    """Remove non-runtime files from a managed Hugging Face materialization directory."""
    target_dir = Path(model_dir).expanduser().resolve()
    if not target_dir.exists():
        return ()
    if not target_dir.is_dir():
        raise ValueError(
            f"Managed Hugging Face materialization path is not a directory: {target_dir}"
        )

    removed_paths: list[Path] = []
    managed_entries = sorted(
        (path for path in target_dir.rglob("*") if path.is_file() or path.is_symlink()),
        key=lambda path: len(path.relative_to(target_dir).parts),
        reverse=True,
    )
    for path in managed_entries:
        relative_path = PurePosixPath(path.relative_to(target_dir).as_posix())
        if _is_hf_runtime_artifact(relative_path):
            continue
        path.unlink()
        removed_paths.append(path)

    empty_dirs = sorted(
        (path for path in target_dir.rglob("*") if path.is_dir()),
        key=lambda path: len(path.relative_to(target_dir).parts),
        reverse=True,
    )
    for directory in empty_dirs:
        if any(directory.iterdir()):
            continue
        directory.rmdir()

    if removed_paths:
        LOGGER.info(
            "Removed %d non-runtime files from %s.",
            len(removed_paths),
            target_dir,
        )
    return tuple(removed_paths)


def hf_runtime_artifacts_complete(model_dir: str | Path) -> bool:
    """Return whether a managed Hugging Face model directory contains the runtime floor."""
    target_dir = Path(model_dir).expanduser().resolve()
    return not _runtime_artifact_gaps(target_dir)


def _has_tokenizer_runtime_artifacts(target_dir: Path) -> bool:
    if any(
        (target_dir / file_name).exists() for file_name in HF_TOKENIZER_RUNTIME_FILES
    ):
        return True
    return all(
        (target_dir / file_name).exists() for file_name in HF_TOKENIZER_PAIR_FILES
    )


def _model_requires_processor_assets(target_dir: Path) -> bool:
    try:
        config = AutoConfig.from_pretrained(target_dir, trust_remote_code=False)
    except Exception:
        return False
    if getattr(config, "vision_config", None) is not None:
        return True
    if getattr(config, "audio_config", None) is not None:
        return True
    try:
        config_dict = config.to_dict()
    except Exception:
        return False
    return "vision_config" in config_dict or "audio_config" in config_dict


def _runtime_artifact_gaps(target_dir: Path) -> tuple[str, ...]:
    gaps: list[str] = []
    if not target_dir.exists():
        return (f"model directory does not exist: {target_dir}",)
    if not target_dir.is_dir():
        return (f"managed model path is not a directory: {target_dir}",)
    if not (target_dir / "config.json").exists():
        gaps.append("missing config.json")
    weight_files = tuple(sorted(target_dir.glob("*.safetensors")))
    if not weight_files:
        gaps.append("missing .safetensors weight files")
    else:
        gaps.extend(_indexed_safetensor_gaps(target_dir))
    if not _has_tokenizer_runtime_artifacts(target_dir):
        gaps.append("missing tokenizer runtime artifacts")
    if _model_requires_processor_assets(target_dir) and not any(
        (target_dir / file_name).exists() for file_name in HF_PROCESSOR_RUNTIME_FILES
    ):
        gaps.append("missing processor runtime artifacts")
    return tuple(gaps)


def _indexed_safetensor_gaps(target_dir: Path) -> tuple[str, ...]:
    gaps: list[str] = []
    target_root = target_dir.resolve()
    for index_path in sorted(target_dir.glob("*.safetensors.index.json")):
        try:
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            gaps.append(f"invalid or unreadable index file: {index_path.name}")
            continue
        weight_map = index_payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            gaps.append(f"missing non-empty weight_map in {index_path.name}")
            continue
        shard_names = set()
        invalid_shard_values = False
        for shard_name in weight_map.values():
            if not isinstance(shard_name, str) or not shard_name:
                invalid_shard_values = True
                continue
            shard_names.add(shard_name)
        if invalid_shard_values:
            gaps.append(f"invalid shard names in {index_path.name}")
        for shard_name in sorted(shard_names):
            shard_path = (target_dir / shard_name).resolve()
            if not shard_path.is_relative_to(target_root):
                gaps.append(
                    f"out-of-root shard path referenced by {index_path.name}: {shard_name}"
                )
                continue
            if not shard_path.is_file():
                gaps.append(
                    f"missing shard referenced by {index_path.name}: {shard_name}"
                )
                continue
            if shard_path.stat().st_size <= 0:
                gaps.append(
                    f"empty shard referenced by {index_path.name}: {shard_name}"
                )
    return tuple(gaps)


def download_hf_snapshot(
    repo_id: str,
    model_dir: str,
    force_download: bool = False,
    revision: str | None = None,
) -> None:
    """Download only runtime-critical Hugging Face artifacts into a local model directory."""
    target_dir = Path(model_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    prune_hf_runtime_artifacts(target_dir)
    LOGGER.info("Downloading runtime artifacts for %s.", repo_id)
    snapshot_download_fn = cast(_SnapshotDownloadCallable, snapshot_download)
    try:
        snapshot_download_fn(
            repo_id=repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            force_download=force_download,
            revision=revision,
            allow_patterns=list(HF_RUNTIME_ARTIFACT_PATTERNS),
        )
    except Exception as exc:
        prune_hf_runtime_artifacts(target_dir)
        artifact_gaps = _runtime_artifact_gaps(target_dir)
        if not artifact_gaps:
            LOGGER.warning(
                "Hugging Face download for %s raised %s after materialization completed.",
                repo_id,
                type(exc).__name__,
            )
            return
        raise ManagedModelDownloadError(
            repo_id,
            target_dir,
            artifact_gaps,
        ) from exc
    prune_hf_runtime_artifacts(target_dir)
    artifact_gaps = _runtime_artifact_gaps(target_dir)
    if artifact_gaps:
        raise ManagedModelDownloadError(repo_id, target_dir, artifact_gaps)
