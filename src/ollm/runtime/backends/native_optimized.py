from ollm.inference import Inference
from ollm.runtime.backends.base import BackendRuntime, ExecutionBackend
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.plan import RuntimePlan


class NativeOptimizedBackend(ExecutionBackend):
    backend_id = "optimized-native"

    def load(self, plan: RuntimePlan, config: RuntimeConfig) -> BackendRuntime:
        if plan.model_path is None or plan.resolved_model.catalog_entry is None:
            raise ValueError("optimized-native backend requires a materialized built-in model alias")

        inference = Inference(
            plan.resolved_model.catalog_entry.model_id,
            device=config.device,
            logging=config.stats or config.verbose,
            multimodality=config.multimodal,
        )
        inference.load_model(str(plan.model_path))
        return BackendRuntime(
            backend_id=self.backend_id,
            model=inference.model,
            tokenizer=inference.tokenizer,
            processor=getattr(inference, "processor", None),
            device=inference.device,
            stats=inference.stats,
            create_cache=lambda cache_dir: inference.DiskCache(cache_dir=str(cache_dir)),
            apply_offload=lambda runtime_config: _apply_native_offload(inference, runtime_config),
        )


def _apply_native_offload(inference: Inference, config: RuntimeConfig) -> None:
    if config.offload_gpu_layers > 0:
        inference.offload_layers_to_gpu_cpu(
            gpu_layers_num=config.offload_gpu_layers,
            cpu_layers_num=config.offload_cpu_layers,
        )
        return
    if config.offload_cpu_layers > 0:
        inference.offload_layers_to_cpu(layers_num=config.offload_cpu_layers)
