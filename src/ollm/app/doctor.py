from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoTokenizer

from ollm.runtime.backend_selector import BackendSelector
from ollm.runtime.capabilities import SupportLevel
from ollm.runtime.config import RuntimeConfig
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

    def __init__(self, resolver: ModelResolver | None = None, selector: BackendSelector | None = None):
        self._resolver = resolver or ModelResolver()
        self._selector = selector or BackendSelector()

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
        cuda_available = torch.cuda.is_available()
        device_checks.append(
            DoctorCheck(
                name="runtime:cuda",
                ok=True,
                message=f"CUDA available: {cuda_available}",
                details={"device": runtime_config.device},
            )
        )
        mps_backend = getattr(torch.backends, "mps", None)
        mps_available = bool(mps_backend is not None and mps_backend.is_available())
        device_checks.append(
            DoctorCheck(
                name="runtime:mps",
                ok=True,
                message=f"MPS available: {mps_available}",
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

    def _check_paths(self, runtime_config: RuntimeConfig) -> list[DoctorCheck]:
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
        resolved_model = self._resolver.resolve(runtime_config.model_reference, runtime_config.resolved_models_dir())
        execution_model = resolved_model
        runtime_plan = None
        if resolved_model.model_path is not None and resolved_model.model_path.exists():
            execution_model = self._resolver.inspect_materialized_model(
                resolved_model.reference,
                resolved_model.model_path,
                source_kind=resolved_model.source_kind,
                repo_id=resolved_model.repo_id,
                revision=resolved_model.revision,
                provider_name=resolved_model.provider_name,
                catalog_entry=resolved_model.catalog_entry,
            )
            runtime_plan = self._selector.select(execution_model, runtime_config)

        resolution_ok = False
        resolution_message = resolved_model.resolution_message
        if resolved_model.source_kind is ModelSourceKind.PROVIDER:
            resolution_message = f"Provider-backed model references are not executable yet: {resolved_model.reference.raw}"
        elif runtime_plan is not None:
            resolution_ok = runtime_plan.is_executable()
            resolution_message = runtime_plan.reason if not resolution_ok else execution_model.resolution_message
        elif resolved_model.is_downloadable():
            resolution_ok = True
        elif resolved_model.capabilities.support_level is not SupportLevel.UNSUPPORTED:
            resolution_ok = True

        checks: list[DoctorCheck] = [
            DoctorCheck(
                name="model:resolution",
                ok=resolution_ok,
                message=resolution_message,
                details={
                    "source_kind": resolved_model.source_kind.value,
                    "support_level": resolved_model.capabilities.support_level.value,
                    "backend_id": None if runtime_plan is None else str(runtime_plan.backend_id),
                },
            )
        ]

        if resolved_model.model_path is None:
            checks.append(
                DoctorCheck(
                    name="model:path",
                    ok=False,
                    message="Model reference does not resolve to a local path",
                    details={"model_reference": runtime_config.model_reference},
                )
            )
            return checks

        installed = resolved_model.model_path.exists()
        checks.append(
            DoctorCheck(
                name="model:path",
                ok=installed,
                message=f"Model path {'exists' if installed else 'does not exist'}",
                details={
                    "path": str(resolved_model.model_path),
                    "model_reference": runtime_config.model_reference,
                },
            )
        )
        if not installed:
            return checks

        try:
            AutoTokenizer.from_pretrained(resolved_model.model_path)
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

        if resolved_model.capabilities.requires_processor:
            try:
                AutoProcessor.from_pretrained(resolved_model.model_path)
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
        resolved_model = self._resolver.resolve(runtime_config.model_reference, runtime_config.resolved_models_dir())
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
