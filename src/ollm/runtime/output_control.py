import io
import logging
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from threading import RLock
from types import ModuleType

_PRINT_SUPPRESSION_LOCK = RLock()
_EXTERNAL_NOISE_LOCK = RLock()
_DEFAULT_EXTERNAL_NOISE_LOGGERS = (
    "transformers",
    "huggingface_hub",
    "ollm.inference",
    "ollm.runtime.specialization.providers",
)


def _silent_print(*args: object, **kwargs: object) -> None:
    del args, kwargs


@contextmanager
def suppress_module_prints(modules: tuple[ModuleType, ...]) -> Iterator[None]:
    if not modules:
        yield
        return

    with _PRINT_SUPPRESSION_LOCK:
        originals: list[tuple[ModuleType, bool, object | None]] = []
        for module in modules:
            had_print = "print" in module.__dict__
            original_print = module.__dict__.get("print")
            originals.append((module, had_print, original_print))
            module.__dict__["print"] = _silent_print
        try:
            yield
        finally:
            for module, had_print, original_print in originals:
                if had_print:
                    module.__dict__["print"] = original_print
                else:
                    module.__dict__.pop("print", None)


@contextmanager
def suppress_external_runtime_noise(
    enabled: bool,
    logger_names: tuple[str, ...] = _DEFAULT_EXTERNAL_NOISE_LOGGERS,
) -> Iterator[None]:
    if not enabled:
        yield
        return

    with _EXTERNAL_NOISE_LOCK:
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            stack.enter_context(_suppress_logger_output(logger_names))
            yield


@contextmanager
def _suppress_logger_output(logger_names: tuple[str, ...]) -> Iterator[None]:
    originals: list[tuple[logging.Logger, int]] = []
    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        originals.append((logger, logger.level))
        logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for logger, original_level in originals:
            logger.setLevel(original_level)
