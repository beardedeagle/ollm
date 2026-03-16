from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from transformers import AutoConfig

from ollm.runtime.capabilities import (
    CapabilityProfile,
    SupportLevel,
    generic_capabilities,
)
from ollm.runtime.catalog import ModelModality


class GenericModelKind(str, Enum):
    CAUSAL_LM = "causal-lm"
    IMAGE_TEXT_TO_TEXT = "image-text-to-text"
    SEQ2SEQ_LM = "seq2seq-lm"


@dataclass(frozen=True, slots=True)
class ModelInspection:
    architecture: str | None
    model_type: str | None
    generic_model_kind: GenericModelKind | None
    capabilities: CapabilityProfile
    requires_trust_remote_code: bool
    message: str


class CapabilityDiscovery:
    def inspect_model_path(self, model_path: Path) -> ModelInspection:
        resolved_path = model_path.expanduser().resolve()
        if not resolved_path.exists() or not resolved_path.is_dir():
            message = f"Resolved local model path '{resolved_path}' does not exist."
            return ModelInspection(
                architecture=None,
                model_type=None,
                generic_model_kind=None,
                capabilities=_unsupported_capabilities(message),
                requires_trust_remote_code=False,
                message=message,
            )

        try:
            config = AutoConfig.from_pretrained(resolved_path, trust_remote_code=False)
        except ValueError as exc:
            message = str(exc)
            requires_trust_remote_code = "trust_remote_code" in message
            reason = (
                "The model requires trust_remote_code=True, which the generic runtime intentionally refuses."
                if requires_trust_remote_code
                else f"Failed to load model config from '{resolved_path}': {message}"
            )
            return ModelInspection(
                architecture=None,
                model_type=None,
                generic_model_kind=None,
                capabilities=_unsupported_capabilities(reason),
                requires_trust_remote_code=requires_trust_remote_code,
                message=reason,
            )
        except Exception as exc:
            reason = f"Failed to load model config from '{resolved_path}': {exc}"
            return ModelInspection(
                architecture=None,
                model_type=None,
                generic_model_kind=None,
                capabilities=_unsupported_capabilities(reason),
                requires_trust_remote_code=False,
                message=reason,
            )

        architecture = _first_architecture(config)
        model_type = _safe_model_type(config)
        has_processor = _has_processor_files(resolved_path)
        has_vision = _has_config_section(config, "vision_config")
        has_audio = _has_config_section(config, "audio_config")
        generic_model_kind = _infer_generic_model_kind(
            config, architecture, has_processor, has_vision, has_audio
        )
        capabilities = _build_capabilities(
            architecture=architecture,
            model_type=model_type,
            generic_model_kind=generic_model_kind,
            has_processor=has_processor,
            has_vision=has_vision,
            has_audio=has_audio,
        )
        message = _build_message(
            resolved_path, architecture, generic_model_kind, capabilities
        )
        return ModelInspection(
            architecture=architecture,
            model_type=model_type,
            generic_model_kind=generic_model_kind,
            capabilities=capabilities,
            requires_trust_remote_code=False,
            message=message,
        )


def _first_architecture(config) -> str | None:
    architectures = getattr(config, "architectures", None) or ()
    if not architectures:
        return None
    return str(architectures[0])


def _safe_model_type(config) -> str | None:
    model_type = getattr(config, "model_type", None)
    if model_type is None:
        return None
    return str(model_type)


def _has_processor_files(model_path: Path) -> bool:
    for file_name in ("processor_config.json", "preprocessor_config.json"):
        if (model_path / file_name).exists():
            return True
    return False


def _has_config_section(config, attribute_name: str) -> bool:
    attribute = getattr(config, attribute_name, None)
    if attribute is not None:
        return True
    try:
        return attribute_name in config.to_dict()
    except Exception:
        return False


def _infer_generic_model_kind(
    config,
    architecture: str | None,
    has_processor: bool,
    has_vision: bool,
    has_audio: bool,
) -> GenericModelKind | None:
    if architecture is not None and architecture.endswith("ForCausalLM"):
        return GenericModelKind.CAUSAL_LM
    if architecture is not None and architecture.endswith("ForConditionalGeneration"):
        if has_audio and not has_vision:
            return None
        if has_vision or has_processor:
            return GenericModelKind.IMAGE_TEXT_TO_TEXT
    if getattr(config, "is_encoder_decoder", False):
        return GenericModelKind.SEQ2SEQ_LM
    return None


def _build_capabilities(
    *,
    architecture: str | None,
    model_type: str | None,
    generic_model_kind: GenericModelKind | None,
    has_processor: bool,
    has_vision: bool,
    has_audio: bool,
) -> CapabilityProfile:
    details: dict[str, str] = {"source": "local-path"}
    if architecture is not None:
        details["architecture"] = architecture
    if model_type is not None:
        details["model_type"] = model_type
    if generic_model_kind is not None:
        details["generic_model_kind"] = generic_model_kind.value

    if generic_model_kind is GenericModelKind.CAUSAL_LM:
        return generic_capabilities(
            modalities=(ModelModality.TEXT,),
            requires_processor=False,
            supports_disk_cache=False,
            details=details,
        )

    if generic_model_kind is GenericModelKind.SEQ2SEQ_LM:
        return generic_capabilities(
            modalities=(ModelModality.TEXT,),
            requires_processor=False,
            supports_disk_cache=False,
            details=details,
        )

    if generic_model_kind is GenericModelKind.IMAGE_TEXT_TO_TEXT:
        modalities = [ModelModality.TEXT]
        if has_vision:
            modalities.append(ModelModality.IMAGE)
        if has_audio:
            modalities.append(ModelModality.AUDIO)
        return generic_capabilities(
            modalities=tuple(modalities),
            requires_processor=has_processor or has_vision or has_audio,
            supports_disk_cache=False,
            details=details,
        )

    if has_audio:
        reason = "Audio generative architectures are not executable through the current generic backend."
    elif architecture is None:
        reason = "Model architecture could not be determined from the local path."
    else:
        reason = f"Architecture '{architecture}' is not executable through the current generic backend."
    details["reason"] = reason
    return CapabilityProfile(
        support_level=SupportLevel.UNSUPPORTED,
        modalities=_unsupported_modalities(has_vision, has_audio),
        requires_processor=has_processor,
        supports_disk_cache=False,
        supports_local_materialization=True,
        supports_specialization=False,
        details=details,
    )


def _unsupported_modalities(
    has_vision: bool, has_audio: bool
) -> tuple[ModelModality, ...]:
    modalities = [ModelModality.TEXT]
    if has_vision:
        modalities.append(ModelModality.IMAGE)
    if has_audio:
        modalities.append(ModelModality.AUDIO)
    return tuple(modalities)


def _unsupported_capabilities(reason: str) -> CapabilityProfile:
    return CapabilityProfile(
        support_level=SupportLevel.UNSUPPORTED,
        modalities=(ModelModality.TEXT,),
        requires_processor=False,
        supports_disk_cache=False,
        supports_local_materialization=True,
        supports_specialization=False,
        details={"reason": reason},
    )


def _build_message(
    model_path: Path,
    architecture: str | None,
    generic_model_kind: GenericModelKind | None,
    capabilities: CapabilityProfile,
) -> str:
    if capabilities.support_level is SupportLevel.UNSUPPORTED:
        reason = capabilities.details.get(
            "reason", "The model is not executable through the current backend set."
        )
        return f"Local model directory '{model_path}' is unsupported: {reason}"
    if architecture is None:
        return f"Local model directory '{model_path}' is executable through the generic backend."
    if generic_model_kind is None:
        return f"Local model directory '{model_path}' with detected architecture '{architecture}'."
    return (
        f"Local model directory '{model_path}' with detected architecture '{architecture}' "
        f"is executable through the {generic_model_kind.value} generic backend."
    )
