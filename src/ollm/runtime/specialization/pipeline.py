from ollm.runtime.config import RuntimeConfig
from ollm.runtime.resolver import ResolvedModel
from ollm.runtime.specialization.base import PlannedSpecialization
from ollm.runtime.specialization.matchers import (
    SpecializationPassMatcher,
    build_default_pass_matchers,
)
from ollm.runtime.specialization.passes import get_specialization_pass
from ollm.runtime.specialization.passes.base import (
    SpecializationPass,
    SpecializationPassId,
)


class SpecializationPipeline:
    def __init__(self, matchers: tuple[SpecializationPassMatcher, ...] | None = None):
        self._matchers = build_default_pass_matchers() if matchers is None else matchers

    def plan(
        self,
        resolved_model: ResolvedModel,
        config: RuntimeConfig,
        provider_id: str | None,
    ) -> PlannedSpecialization:
        if provider_id is None:
            return PlannedSpecialization(provider_id=None)

        selected_passes: list[SpecializationPass] = []
        seen_pass_ids: set[SpecializationPassId] = set()
        for matcher in self._matchers:
            if not matcher.matches(resolved_model, config, provider_id):
                continue
            specialization_pass = get_specialization_pass(matcher.pass_id)
            if specialization_pass.pass_id in seen_pass_ids:
                continue
            seen_pass_ids.add(specialization_pass.pass_id)
            selected_passes.append(specialization_pass)

        details = {
            "provider_id": provider_id,
            "pass_count": str(len(selected_passes)),
        }
        if resolved_model.native_family is not None:
            details["native_family"] = resolved_model.native_family.value
        if resolved_model.architecture is not None:
            details["architecture"] = resolved_model.architecture
        return PlannedSpecialization(
            provider_id=provider_id,
            passes=tuple(selected_passes),
            details=details,
        )
