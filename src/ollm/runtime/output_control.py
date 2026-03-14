from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock
from types import ModuleType

_PRINT_SUPPRESSION_LOCK = RLock()


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
