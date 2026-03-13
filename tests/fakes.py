from dataclasses import dataclass
from pathlib import Path

from ollm.app.doctor import DoctorCheck, DoctorReport
from ollm.app.types import Message, PromptResponse
from ollm.runtime.config import RuntimeConfig


@dataclass(slots=True)
class FakeLoadedRuntime:
    config: RuntimeConfig


class FakeRuntimeLoader:
    def __init__(self):
        self.load_calls: list[str] = []
        self.download_calls: list[tuple[str, Path, bool]] = []

    def load(self, config: RuntimeConfig) -> FakeLoadedRuntime:
        self.load_calls.append(config.model_id)
        return FakeLoadedRuntime(config=config)

    def download(self, model_id: str, models_dir: Path, force_download: bool = False) -> Path:
        self.download_calls.append((model_id, models_dir, force_download))
        target = models_dir / model_id
        target.mkdir(parents=True, exist_ok=True)
        return target


class FakeRuntimeExecutor:
    def __init__(self):
        self.prompts: list[str] = []

    def execute(self, runtime, request, sink=None) -> PromptResponse:
        del runtime
        prompt_text = request.messages[-1].text_content()
        self.prompts.append(prompt_text)
        text = f"echo:{prompt_text}"
        if sink is not None:
            sink.on_status("fake")
            sink.on_text(text)
            sink.on_complete(text)
        return PromptResponse(text=text, assistant_message=Message.assistant_text(text), metadata={})


class FakeDoctorService:
    def run(self, runtime_config, include_imports=True, include_runtime=True, include_paths=True, include_download=False):
        del runtime_config, include_imports, include_runtime, include_paths, include_download
        return DoctorReport([DoctorCheck(name="doctor:fake", ok=True, message="ok")])

