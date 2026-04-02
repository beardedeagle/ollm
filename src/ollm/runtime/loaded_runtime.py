"""Loaded runtime bundle helpers shared across runtime execution surfaces."""

from dataclasses import dataclass, field
from pathlib import Path

import torch

from ollm.runtime.backends.base import BackendRuntime
from ollm.runtime.capabilities import CapabilityProfile
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.plan import RuntimePlan
from ollm.runtime.resolver import ResolvedModel


@dataclass(slots=True)
class LoadedRuntime:
    """Loaded runtime bundle containing the finalized backend and plan metadata.

    Attributes:
        resolved_model (ResolvedModel): Final resolved model metadata for the
            loaded runtime.
        config (RuntimeConfig): Effective runtime configuration after selector
            application.
        backend (BackendRuntime): Loaded backend runtime implementation.
        model_path (Path | None): Local materialized model path when one exists.
        plan (RuntimePlan): Final runtime plan used to load the backend.
    """

    resolved_model: ResolvedModel
    config: RuntimeConfig
    backend: BackendRuntime
    model_path: Path | None
    plan: RuntimePlan
    _kv_cache_instances: dict[tuple[Path, str, str, int | None], object] = field(
        default_factory=dict
    )

    @property
    def capabilities(self) -> CapabilityProfile:
        """Return capability information aligned with the finalized runtime plan.

        Returns:
            CapabilityProfile: Capability metadata adjusted to reflect the final
            support level and disk-cache behavior of the loaded runtime.
        """
        resolved_capabilities = self.resolved_model.capabilities
        if (
            resolved_capabilities.support_level is self.plan.support_level
            and resolved_capabilities.supports_disk_cache
            is self.plan.supports_disk_cache
        ):
            return resolved_capabilities
        details = dict(resolved_capabilities.details)
        details["runtime_plan_support_level"] = self.plan.support_level.value
        return CapabilityProfile(
            support_level=self.plan.support_level,
            modalities=resolved_capabilities.modalities,
            requires_processor=resolved_capabilities.requires_processor,
            supports_disk_cache=self.plan.supports_disk_cache,
            supports_local_materialization=resolved_capabilities.supports_local_materialization,
            supports_specialization=resolved_capabilities.supports_specialization,
            details=details,
        )

    @property
    def model(self):
        """Expose the backend-owned model object when one exists."""
        return self.backend.model

    @property
    def tokenizer(self):
        """Expose the backend-owned tokenizer when one exists."""
        return self.backend.tokenizer

    @property
    def processor(self):
        """Expose the backend-owned processor when one exists."""
        return self.backend.processor

    @property
    def device(self) -> torch.device:
        """Expose the backend runtime device."""
        return self.backend.device

    def get_or_create_kv_cache(
        self,
        cache_dir: Path,
        strategy: str,
        lifecycle: str,
        window_tokens: int | None,
    ) -> object | None:
        """Reuse one KV-cache instance per resolved cache key.

        Args:
            cache_dir (Path): Cache root for the KV cache instance.
            strategy (str): Resolved KV cache strategy ID.
            lifecycle (str): Resolved cache lifecycle ID.
            window_tokens (int | None): Sliding-window token budget when the
                strategy requires one.

        Returns:
            object | None: Existing or newly created cache object, or ``None``
            when the backend does not expose a cache.
        """
        cache_key = (cache_dir.resolve(), strategy, lifecycle, window_tokens)
        cache = self._kv_cache_instances.get(cache_key)
        if cache is not None:
            return cache
        created_cache = self.backend.create_cache(
            cache_dir, strategy, lifecycle, window_tokens
        )
        if created_cache is not None:
            self._kv_cache_instances[cache_key] = created_cache
        return created_cache

    def reset_kv_cache_instances(self) -> None:
        """Drop any cached KV objects before a full-history re-execution."""
        self._kv_cache_instances.clear()
