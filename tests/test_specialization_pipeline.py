from pathlib import Path

from ollm.runtime.capabilities import CapabilityProfile, SupportLevel
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.catalog import ModelModality
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.reference import ModelReference
from ollm.runtime.resolver import ModelSourceKind, NativeFamily, ResolvedModel
from ollm.runtime.specialization import SpecializationPipeline
from ollm.runtime.specialization.passes.base import SpecializationPassId


def _build_resolved_model(
	model_path: Path,
	*,
	native_family: NativeFamily,
	architecture: str,
	modalities: tuple[ModelModality, ...] = (ModelModality.TEXT,),
	supports_disk_cache: bool = True,
) -> ResolvedModel:
	capabilities = CapabilityProfile(
		support_level=SupportLevel.OPTIMIZED,
		modalities=modalities,
		requires_processor=ModelModality.IMAGE in modalities or ModelModality.AUDIO in modalities,
		supports_disk_cache=supports_disk_cache,
		supports_specialization=True,
	)
	return ResolvedModel(
		reference=ModelReference.parse(str(model_path)),
		source_kind=ModelSourceKind.LOCAL_PATH,
		normalized_name=model_path.name,
		model_path=model_path,
		repo_id=None,
		revision=None,
		provider_name=None,
		catalog_entry=None,
		capabilities=capabilities,
		native_family=native_family,
		resolution_message="materialized",
		architecture=architecture,
		model_type=native_family.value,
		generic_model_kind=GenericModelKind.CAUSAL_LM,
	)


def test_specialization_pipeline_selects_qwen3_next_passes(tmp_path: Path) -> None:
	model_path = tmp_path / "qwen3-next"
	model_path.mkdir()
	planned_specialization = SpecializationPipeline().plan(
		_build_resolved_model(
			model_path,
			native_family=NativeFamily.QWEN3_NEXT,
			architecture="Qwen3MoeForCausalLM",
		),
		RuntimeConfig(device="cpu"),
		"qwen3-next-native",
	)

	assert planned_specialization.pass_ids == (
		SpecializationPassId.DISK_CACHE,
		SpecializationPassId.CPU_OFFLOAD,
		SpecializationPassId.GPU_OFFLOAD,
		SpecializationPassId.MOE_ROUTING,
	)
	assert planned_specialization.traits.supports_disk_cache is True
	assert planned_specialization.traits.supports_cpu_offload is True
	assert planned_specialization.traits.supports_gpu_offload is True


def test_specialization_pipeline_selects_multimodal_shell_only_when_enabled(tmp_path: Path) -> None:
	model_path = tmp_path / "gemma3"
	model_path.mkdir()
	resolved_model = _build_resolved_model(
		model_path,
		native_family=NativeFamily.GEMMA3,
		architecture="Gemma3ForConditionalGeneration",
		modalities=(ModelModality.TEXT, ModelModality.IMAGE),
	)
	enabled_plan = SpecializationPipeline().plan(
		resolved_model,
		RuntimeConfig(device="cpu", multimodal=True),
		"gemma3-native",
	)
	disabled_plan = SpecializationPipeline().plan(
		resolved_model,
		RuntimeConfig(device="cpu", multimodal=False),
		"gemma3-native",
	)

	assert SpecializationPassId.MULTIMODAL_SHELL in enabled_plan.pass_ids
	assert SpecializationPassId.MULTIMODAL_SHELL not in disabled_plan.pass_ids


def test_specialization_pipeline_returns_empty_plan_without_provider() -> None:
	planned_specialization = SpecializationPipeline().plan(
		ResolvedModel(
			reference=ModelReference.parse("Qwen/Qwen2.5-7B-Instruct"),
			source_kind=ModelSourceKind.HUGGING_FACE,
			normalized_name="Qwen/Qwen2.5-7B-Instruct",
			model_path=None,
			repo_id="Qwen/Qwen2.5-7B-Instruct",
			revision=None,
			provider_name=None,
			catalog_entry=None,
			capabilities=CapabilityProfile(
				support_level=SupportLevel.GENERIC,
				supports_specialization=False,
			),
			native_family=None,
			resolution_message="generic",
			architecture=None,
			model_type=None,
			generic_model_kind=None,
		),
		RuntimeConfig(device="cpu"),
		None,
	)

	assert planned_specialization.provider_id is None
	assert planned_specialization.pass_ids == ()
