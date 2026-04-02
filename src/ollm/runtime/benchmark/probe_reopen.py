"""Reopen-session-growth probe entrypoint."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from ollm.app.types import ContentPart, Message, MessageRole
from ollm.client import RuntimeClient
from ollm.runtime.benchmark.probe_execution import (
    _clear_backend_stats,
    build_prompt_request,
    execute_request_probe,
)
from ollm.runtime.benchmark.probe_types import (
    ReopenSessionGrowthProbeResult,
    ReopenSessionGrowthTurn,
)
from ollm.runtime.benchmark.resources import measure_stage
from ollm.runtime.config import DEFAULT_SYSTEM_PROMPT, GenerationConfig, RuntimeConfig
from ollm.runtime.loaded_runtime import LoadedRuntime
from ollm.runtime.strategy_selector import DEFAULT_STRATEGY_SELECTOR_PROFILE


def run_reopen_session_growth_probe(
    *,
    model_reference: str,
    models_dir: Path,
    device: str,
    backend: str,
    use_specialization: bool,
    session_turns: int,
    max_new_tokens: int,
    kv_cache_strategy: str | None,
    strategy_selector_profile: str = DEFAULT_STRATEGY_SELECTOR_PROFILE,
    kv_cache_window_tokens: int | None,
    offload_cpu_layers: int,
    offload_cpu_policy: str,
    offload_gpu_layers: int,
) -> ReopenSessionGrowthProbeResult:
    generation_config = GenerationConfig(
        stream=True,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    history: list[Message] = []
    turns: list[ReopenSessionGrowthTurn] = []
    with TemporaryDirectory(prefix="ollm-reopen-session-growth-") as temp_dir:
        persistent_cache_dir = Path(temp_dir)
        for turn_index in range(1, session_turns + 1):
            runtime_config = RuntimeConfig(
                model_reference=model_reference,
                models_dir=models_dir.expanduser().resolve(),
                device=device,
                backend=backend,
                use_specialization=use_specialization,
                cache_dir=persistent_cache_dir,
                use_cache=True,
                kv_cache_strategy=kv_cache_strategy,
                strategy_selector_profile=strategy_selector_profile,
                kv_cache_lifecycle="persistent",
                kv_cache_window_tokens=kv_cache_window_tokens,
                offload_cpu_layers=offload_cpu_layers,
                offload_cpu_policy=offload_cpu_policy,
                offload_gpu_layers=offload_gpu_layers,
                stats=True,
            )
            client = RuntimeClient()
            runtime_result = measure_stage(
                runtime_config.device,
                lambda: client.load(runtime_config),
            )
            runtime = cast(LoadedRuntime, runtime_result[0])
            _clear_backend_stats(runtime)
            user_message = Message(
                role=MessageRole.USER,
                content=[
                    ContentPart.text(
                        f"Turn {turn_index}: summarize the benchmark status in one sentence."
                    )
                ],
            )
            execution = execute_request_probe(
                runtime=runtime,
                request=build_prompt_request(
                    runtime_config=runtime.config,
                    generation_config=generation_config,
                    messages=[
                        Message(
                            role=MessageRole.SYSTEM,
                            content=[ContentPart.text(DEFAULT_SYSTEM_PROMPT)],
                        ),
                        *history,
                        user_message,
                    ],
                ),
            )
            history.append(user_message)
            history.append(Message.assistant_text(execution.response_text))
            turns.append(
                ReopenSessionGrowthTurn(
                    turn_index=turn_index,
                    runtime_load_ms=runtime_result[1],
                    runtime_load_resources=runtime_result[2],
                    request=execution.metrics,
                )
            )
    return ReopenSessionGrowthProbeResult(turns=tuple(turns))
