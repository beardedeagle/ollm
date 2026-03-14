from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoTokenizer

from ollm.runtime.backend_selector import BackendSelector
from ollm.runtime.capabilities import SupportLevel
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.loader import RuntimeLoader
from ollm.runtime.resolver import ModelResolver, ModelSourceKind


@dataclass(slots=True)
class DoctorCheck:
    name: str
    ok: bool
    message: str
    details: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "ok": self.ok,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(slots=True)
class DoctorReport:
    checks: list[DoctorCheck]

    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok(),
            "checks": [check.as_dict() for check in self.checks],
        }


class DoctorService:
    CORE_IMPORTS = ("ollm", "torch", "transformers", "huggingface_hub")
    OPTIONAL_IMPORTS = {
        "adapters": ("peft",),
        "audio": ("librosa", "mistral_common"),
        "cuda": ("flash_attn", "triton"),
        "export": ("safetensors",),
    }

    def __init__(
        self,
        runtime_loader: RuntimeLoader | None = None,
        resolver: ModelResolver | None = None,
        selector: BackendSelector | None = None,
    ):
        if runtime_loader is None:
            base_resolver = resolver or ModelResolver()
            base_selector = selector or BackendSelector()
            runtime_loader = RuntimeLoader(resolver=base_resolver, selector=base_selector)
        self._runtime_loader = runtime_loader

    def run(
        self,
        runtime_config: RuntimeConfig,
        include_imports: bool = True,
        include_runtime: bool = True,
        include_paths: bool = True,
        include_download: bool = False,
    ) -> DoctorReport:
        checks: list[DoctorCheck] = []
        if include_imports:
            checks.extend(self._check_imports())
        if include_runtime:
            checks.extend(self._check_runtime(runtime_config))
        if include_paths:
            checks.extend(self._check_paths(runtime_config))
            checks.extend(self._check_model(runtime_config))
        if include_download:
            checks.append(self._check_download(runtime_config))
        return DoctorReport(checks=checks)

    def _check_imports(self) -> list[DoctorCheck]:
        checks: list[DoctorCheck] = []
        for module_name in self.CORE_IMPORTS:
            checks.append(self._import_check(module_name, optional=False))
        for extra_name, module_names in self.OPTIONAL_IMPORTS.items():
            for module_name in module_names:
                checks.append(self._import_check(module_name, optional=True, extra_name=extra_name))
        return checks

    def _import_check(self, module_name: str, optional: bool, extra_name: str = "") -> DoctorCheck:
        try:
            import_module(module_name)
        except Exception as exc:
            if optional:
                return DoctorCheck(
                    name=f"import:{module_name}",
                    ok=True,
                    message=f"Optional dependency {module_name} is not installed",
                    details={"extra": extra_name, "reason": str(exc)},
                )
            return DoctorCheck(
                name=f"import:{module_name}",
                ok=False,
                message=f"Failed to import {module_name}",
                details={"reason": str(exc)},
            )
        return DoctorCheck(name=f"import:{module_name}", ok=True, message=f"Imported {module_name}")

    def _check_runtime(self, runtime_config: RuntimeConfig) -> list[DoctorCheck]:
        device_checks: list[DoctorCheck] = []
        resolved_model = self._runtime_loader.resolve(
            runtime_config.model_reference,
            runtime_config.resolved_models_dir(),
        )
        if resolved_model.source_kind is ModelSourceKind.PROVIDER:
            device_checks.append(
                DoctorCheck(
                    name="runtime:requested-device",
                    ok=True,
                    message=(
                        f"Provider-backed model references for {resolved_model.provider_name} ignore "
                        f"local device '{runtime_config.device}'."
                    ),
                    details={"provider": "" if resolved_model.provider_name is None else resolved_model.provider_name},
                )
            )
            device_checks.append(
                DoctorCheck(
                    name="runtime:cpu",
                    ok=True,
                    message="CPU runtime available",
                    details={"threads": str(torch.get_num_threads())},
                )
            )
            return device_checks

        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0
        mps_backend = getattr(torch.backends, "mps", None)
        mps_available = bool(mps_backend is not None and mps_backend.is_available())
        requested_device = runtime_config.device.strip()
        requested_backend, _, requested_index = requested_device.partition(":")
        requested_backend = requested_backend.lower()
        if requested_backend == "cuda":
            device_checks.append(
                self._requested_cuda_check(
                    requested_device=requested_device,
                    requested_index=requested_index,
                    cuda_available=cuda_available,
                    cuda_device_count=cuda_device_count,
                )
            )
        elif requested_backend == "mps":
            device_checks.append(
                DoctorCheck(
                    name="runtime:requested-device",
                    ok=mps_available,
                    message=(
                        f"Requested device '{requested_device}' is available."
                        if mps_available
                        else f"Requested device '{requested_device}' is not available."
                    ),
                    details={"mps_available": str(mps_available)},
                )
            )
        else:
            device_checks.append(
                DoctorCheck(
                    name="runtime:requested-device",
                    ok=True,
                    message=f"Requested device '{requested_device}' is available.",
                    details={"backend": requested_backend},
                )
            )
        cpu_threads = torch.get_num_threads()
        device_checks.append(
            DoctorCheck(
                name="runtime:cpu",
                ok=True,
                message="CPU runtime available",
                details={"threads": str(cpu_threads)},
            )
        )
        return device_checks

    def _requested_cuda_check(
        self,
        requested_device: str,
        requested_index: str,
        cuda_available: bool,
        cuda_device_count: int,
    ) -> DoctorCheck:
        if not cuda_available:
            return DoctorCheck(
                name="runtime:requested-device",
                ok=False,
                message=f"Requested device '{requested_device}' is not available.",
                details={
                    "cuda_available": str(cuda_available),
                    "cuda_device_count": str(cuda_device_count),
                },
            )
        if not requested_index:
            return DoctorCheck(
                name="runtime:requested-device",
                ok=True,
                message=f"Requested device '{requested_device}' is available.",
                details={
                    "cuda_available": str(cuda_available),
                    "cuda_device_count": str(cuda_device_count),
                },
            )
        if not requested_index.isdigit():
            return DoctorCheck(
                name="runtime:requested-device",
                ok=False,
                message=f"Requested CUDA device '{requested_device}' has an invalid index.",
                details={
                    "cuda_available": str(cuda_available),
                    "cuda_device_count": str(cuda_device_count),
                },
            )
        device_index = int(requested_index)
        device_available = device_index < cuda_device_count
        return DoctorCheck(
            name="runtime:requested-device",
            ok=device_available,
            message=(
                f"Requested device '{requested_device}' is available."
                if device_available
                else f"Requested device '{requested_device}' is not available."
            ),
            details={
                "cuda_available": str(cuda_available),
                "cuda_device_count": str(cuda_device_count),
            },
        )

    def _check_paths(self, runtime_config: RuntimeConfig) -> list[DoctorCheck]:
        resolved_model = self._runtime_loader.resolve(
            runtime_config.model_reference,
            runtime_config.resolved_models_dir(),
        )
        if resolved_model.source_kind is ModelSourceKind.PROVIDER:
            return [
                DoctorCheck(
                    name="path:models-dir",
                    ok=True,
                    message="Provider-backed model references do not use the local models directory",
                    details={"provider": "" if resolved_model.provider_name is None else resolved_model.provider_name},
                )
            ]
        checks: list[DoctorCheck] = []
        checks.append(self._path_check("models-dir", runtime_config.resolved_models_dir(), create=False))
        if runtime_config.use_cache:
            checks.append(self._path_check("cache-dir", runtime_config.resolved_cache_dir(), create=True))
        adapter_dir = runtime_config.resolved_adapter_dir()
        if adapter_dir is not None:
            checks.append(self._path_check("adapter-dir", adapter_dir, create=False))
        return checks

    def _path_check(self, label: str, path: Path, create: bool) -> DoctorCheck:
        try:
            if create:
                path.mkdir(parents=True, exist_ok=True)
            exists = path.exists()
            writable = False
            if exists and path.is_dir():
                probe = path / '.doctor-write-check'
                probe.write_text('ok', encoding='utf-8')
                probe.unlink()
                writable = True
            return DoctorCheck(
                name=f"path:{label}",
                ok=exists,
                message=f"{label} {'exists' if exists else 'does not exist'}",
                details={"path": str(path), "writable": str(writable)},
            )
        except Exception as exc:
            return DoctorCheck(
                name=f"path:{label}",
                ok=False,
                message=f"Failed to validate {label}",
                details={"path": str(path), "reason": str(exc)},
            )

    def _check_model(self, runtime_config: RuntimeConfig) -> list[DoctorCheck]:
        resolved_model = self._runtime_loader.resolve(
            runtime_config.model_reference,
            runtime_config.resolved_models_dir(),
        )
        runtime_plan = self._runtime_loader.plan(runtime_config)
        execution_model = runtime_plan.resolved_model

        resolution_ok = runtime_plan.is_executable()
        resolution_message = runtime_plan.reason
        if not resolution_ok and execution_model.is_downloadable():
            resolution_ok = True
        elif (
            not resolution_ok
            and execution_model.source_kind is not ModelSourceKind.PROVIDER
            and execution_model.capabilities.support_level is not SupportLevel.UNSUPPORTED
        ):
            resolution_ok = True

        checks: list[DoctorCheck] = [
            DoctorCheck(
                name="model:resolution",
                ok=resolution_ok,
                message=resolution_message,
                details={
                    "source_kind": resolved_model.source_kind.value,
                    "support_level": runtime_plan.support_level.value,
                    "backend_id": "" if runtime_plan.backend_id is None else runtime_plan.backend_id,
                    "specialization_provider_id": (
                        "" if runtime_plan.specialization_provider_id is None else runtime_plan.specialization_provider_id
                    ),
                    "specialization_state": runtime_plan.specialization_state.value,
                    "planned_specialization_pass_ids": ",".join(
                        pass_id.value for pass_id in runtime_plan.specialization_pass_ids
                    ),
                },
            )
        ]

        if execution_model.source_kind is ModelSourceKind.PROVIDER:
            checks.append(
                DoctorCheck(
                    name="model:path",
                    ok=True,
                    message="Provider-backed model references do not use a local materialization path",
                    details={
                        "model_reference": runtime_config.model_reference,
                        "provider": "" if execution_model.provider_name is None else execution_model.provider_name,
                    },
                )
            )
            return checks

        if execution_model.model_path is None:
            checks.append(
                DoctorCheck(
                    name="model:path",
                    ok=False,
                    message="Model reference does not resolve to a local path",
                    details={"model_reference": runtime_config.model_reference},
                )
            )
            return checks

        installed = execution_model.model_path.exists()
        checks.append(
            DoctorCheck(
                name="model:path",
                ok=installed,
                message=f"Model path {'exists' if installed else 'does not exist'}",
                details={
                    "path": str(execution_model.model_path),
                    "model_reference": runtime_config.model_reference,
                },
            )
        )
        if not installed:
            return checks

        try:
            AutoTokenizer.from_pretrained(execution_model.model_path)
            checks.append(DoctorCheck(name="model:tokenizer", ok=True, message="Tokenizer loads successfully"))
        except Exception as exc:
            checks.append(
                DoctorCheck(
                    name="model:tokenizer",
                    ok=False,
                    message="Tokenizer failed to load",
                    details={"reason": str(exc)},
                )
            )

        if execution_model.capabilities.requires_processor:
            try:
                AutoProcessor.from_pretrained(execution_model.model_path)
                checks.append(DoctorCheck(name="model:processor", ok=True, message="Processor loads successfully"))
            except Exception as exc:
                checks.append(
                    DoctorCheck(
                        name="model:processor",
                        ok=False,
                        message="Processor failed to load",
                        details={"reason": str(exc)},
                    )
                )
        return checks

    def _check_download(self, runtime_config: RuntimeConfig) -> DoctorCheck:
        resolved_model = self._runtime_loader.resolve(
            runtime_config.model_reference,
            runtime_config.resolved_models_dir(),
        )
        if not resolved_model.is_downloadable() or resolved_model.repo_id is None or resolved_model.model_path is None:
            return DoctorCheck(
                name="download:ready",
                ok=False,
                message=f"Model reference '{runtime_config.model_reference}' is not downloadable via the native snapshot flow",
                details={"source_kind": resolved_model.source_kind.value},
            )

        models_dir = runtime_config.resolved_models_dir()
        target_path = resolved_model.model_path
        try:
            models_dir.mkdir(parents=True, exist_ok=True)
            probe = models_dir / '.doctor-download-check'
            probe.write_text('ok', encoding='utf-8')
            probe.unlink()
            return DoctorCheck(
                name="download:ready",
                ok=True,
                message=f"Download target is writable for {runtime_config.model_reference}",
                details={"repo_id": resolved_model.repo_id, "target": str(target_path)},
            )
        except Exception as exc:
            return DoctorCheck(
                name="download:ready",
                ok=False,
                message=f"Download target is not writable for {runtime_config.model_reference}",
                details={"repo_id": resolved_model.repo_id, "target": str(target_path), "reason": str(exc)},
            )
