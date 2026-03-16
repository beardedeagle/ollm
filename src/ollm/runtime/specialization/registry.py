from ollm.runtime.config import RuntimeConfig
from ollm.runtime.resolver import ResolvedModel
from ollm.runtime.specialization.base import (
    OptimizedModelArtifacts,
    SpecializationMatch,
    SpecializationProvider,
)
from ollm.utils import Stats


class SpecializationRegistry:
    def __init__(self, providers: tuple[SpecializationProvider, ...]):
        if not providers:
            raise ValueError("SpecializationRegistry requires at least one provider")
        provider_map: dict[str, SpecializationProvider] = {}
        for provider in providers:
            if provider.provider_id in provider_map:
                raise ValueError(
                    f"Duplicate specialization provider id: {provider.provider_id}"
                )
            provider_map[provider.provider_id] = provider
        self._providers = providers
        self._provider_map = provider_map

    def provider_ids(self) -> tuple[str, ...]:
        return tuple(provider.provider_id for provider in self._providers)

    def select(
        self, resolved_model: ResolvedModel, config: RuntimeConfig
    ) -> SpecializationMatch | None:
        matches: list[SpecializationMatch] = []
        for provider in self._providers:
            match = provider.match(resolved_model, config)
            if match is not None:
                matches.append(match)

        if not matches:
            return None
        if len(matches) > 1:
            provider_ids = ", ".join(match.provider_id for match in matches)
            raise ValueError(
                f"Ambiguous specialization providers for {resolved_model.reference.raw}: {provider_ids}"
            )
        return matches[0]

    def load(
        self,
        provider_id: str,
        resolved_model: ResolvedModel,
        config: RuntimeConfig,
        stats: Stats | None,
    ) -> OptimizedModelArtifacts:
        provider = self._provider_map.get(provider_id)
        if provider is None:
            raise ValueError(f"Unknown specialization provider id: {provider_id}")
        return provider.load(resolved_model, config, stats)
