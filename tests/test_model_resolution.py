from pathlib import Path

from ollm.runtime.capabilities import SupportLevel
from ollm.runtime.resolver import ModelResolver, ModelSourceKind


def test_resolver_accepts_built_in_alias() -> None:
    resolved = ModelResolver().resolve("llama3-1B-chat", Path("models"))
    assert resolved.source_kind is ModelSourceKind.BUILTIN
    assert resolved.capabilities.support_level is SupportLevel.OPTIMIZED
    assert resolved.repo_id == "unsloth/Llama-3.2-1B-Instruct"


def test_resolver_accepts_hugging_face_repo_id() -> None:
    resolved = ModelResolver().resolve("Qwen/Qwen2.5-7B-Instruct", Path("models"))
    assert resolved.source_kind is ModelSourceKind.HUGGING_FACE
    assert resolved.capabilities.support_level is SupportLevel.UNSUPPORTED
    assert resolved.repo_id == "Qwen/Qwen2.5-7B-Instruct"


def test_resolver_preserves_hugging_face_revision() -> None:
    resolved = ModelResolver().resolve("hf://Qwen/Qwen2.5-7B-Instruct@main", Path("models"))
    assert resolved.source_kind is ModelSourceKind.HUGGING_FACE
    assert resolved.revision == "main"
    assert resolved.model_path == Path("models").expanduser().resolve() / "Qwen--Qwen2.5-7B-Instruct--main"


def test_resolver_accepts_local_path(tmp_path: Path) -> None:
    model_dir = tmp_path / "my-model"
    model_dir.mkdir()
    resolved = ModelResolver().resolve(str(model_dir), tmp_path)
    assert resolved.source_kind is ModelSourceKind.LOCAL_PATH
    assert resolved.model_path == model_dir.resolve()


def test_resolver_marks_unknown_opaque_reference_as_unsupported() -> None:
    resolved = ModelResolver().resolve("qwen3.5:9b-bf16", Path("models"))
    assert resolved.source_kind is ModelSourceKind.OPAQUE
    assert resolved.capabilities.support_level is SupportLevel.UNSUPPORTED
