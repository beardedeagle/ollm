"""Rule helpers for the repo-local Python standards checker."""

import ast
import io
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path

DISALLOWED_TYPING_NAMES: frozenset[str] = frozenset({"Any", "Optional", "Union"})
SYNC_IO_CALLS: frozenset[tuple[str, str | None]] = frozenset(
    {
        ("open", None),
        ("Path", "open"),
        ("Path", "read_bytes"),
        ("Path", "read_text"),
        ("Path", "write_bytes"),
        ("Path", "write_text"),
        ("torch", "load"),
        ("torch", "save"),
        ("subprocess", "Popen"),
        ("subprocess", "run"),
        ("os", "fdopen"),
        ("os", "makedirs"),
        ("os", "open"),
        ("shutil", "rmtree"),
    }
)
PARTIAL_WORK_MARKER_PATTERN = re.compile(r"\b(TODO|FIXME|XXX)\b")
MACHINE_SPECIFIC_PATH_PREFIXES: tuple[str, ...] = (
    "/content/",
    "/home/",
    "/media/",
)


@dataclass(frozen=True, slots=True)
class Violation:
    rule_id: str
    severity: str
    category: str
    path: Path
    line: int
    column: int
    message: str


class StandardsVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self._path = path
        self.violations: list[Violation] = []

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "__future__":
            for alias in node.names:
                if alias.name == "annotations":
                    self._add(
                        "future-annotations",
                        "error",
                        "mechanical",
                        node.lineno,
                        node.col_offset,
                        "from __future__ import annotations is forbidden",
                    )
        if node.level > 0:
            self._add(
                "relative-import",
                "error",
                "mechanical",
                node.lineno,
                node.col_offset,
                "relative imports are forbidden",
            )
        if node.module == "typing":
            for alias in node.names:
                if alias.name in DISALLOWED_TYPING_NAMES:
                    self._add(
                        "typing-alias",
                        "error",
                        "mechanical",
                        node.lineno,
                        node.col_offset,
                        f"{alias.name} is forbidden; use concrete types and | unions",
                    )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in DISALLOWED_TYPING_NAMES:
            self._add(
                "typing-name",
                "error",
                "mechanical",
                node.lineno,
                node.col_offset,
                f"{node.id} is forbidden; use concrete types and | unions",
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "typing"
            and node.attr in DISALLOWED_TYPING_NAMES
        ):
            self._add(
                "typing-attribute",
                "error",
                "mechanical",
                node.lineno,
                node.col_offset,
                f"typing.{node.attr} is forbidden; use concrete types and | unions",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        call_target = _call_target(node.func)
        if call_target in SYNC_IO_CALLS:
            self._add(
                "sync-io",
                "audit",
                "architectural",
                node.lineno,
                node.col_offset,
                f"synchronous I/O call detected: {_render_call_target(call_target)}",
            )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        _check_annotations(self._path, node, self.violations)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        _check_annotations(self._path, node, self.violations)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        _check_annotation_node(
            path=self._path,
            annotation=node.annotation,
            violations=self.violations,
            line=node.lineno,
            column=node.col_offset,
        )
        self.generic_visit(node)

    def _add(
        self,
        rule_id: str,
        severity: str,
        category: str,
        line: int,
        column: int,
        message: str,
    ) -> None:
        self.violations.append(
            Violation(
                rule_id=rule_id,
                severity=severity,
                category=category,
                path=self._path,
                line=line,
                column=column,
                message=message,
            )
        )


def scan_partial_work_markers(*, path: Path, source: str) -> list[Violation]:
    violations: list[Violation] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        marker_match = PARTIAL_WORK_MARKER_PATTERN.search(token.string)
        if marker_match is None:
            continue
        violations.append(
            Violation(
                rule_id="partial-work-marker",
                severity="error",
                category="mechanical",
                path=path,
                line=token.start[0],
                column=token.start[1] + marker_match.start(),
                message=f"partial-work marker detected: {marker_match.group(1)}",
            )
        )
    return violations


def check_import_placement(
    *, path: Path, tree: ast.Module, violations: list[Violation]
) -> None:
    first_non_import_seen = False
    for index, statement in enumerate(tree.body):
        if index == 0 and _is_module_docstring(statement):
            continue
        if isinstance(statement, ast.Import | ast.ImportFrom):
            if first_non_import_seen:
                violations.append(
                    Violation(
                        rule_id="late-import",
                        severity="error",
                        category="mechanical",
                        path=path,
                        line=statement.lineno,
                        column=statement.col_offset,
                        message="imports must stay at the top of the module",
                    )
                )
            continue
        first_non_import_seen = True


def check_script_top_level_statements(
    *, path: Path, tree: ast.Module, violations: list[Violation]
) -> None:
    """Reject script modules that execute real work at import time."""
    if "scripts" not in path.parts:
        return
    for index, statement in enumerate(tree.body):
        if index == 0 and _is_module_docstring(statement):
            continue
        if isinstance(
            statement,
            ast.Import
            | ast.ImportFrom
            | ast.FunctionDef
            | ast.AsyncFunctionDef
            | ast.ClassDef,
        ):
            continue
        if isinstance(statement, ast.Assign):
            if _assignment_value_is_safe(statement.value):
                continue
        if isinstance(statement, ast.AnnAssign):
            if statement.value is None or _assignment_value_is_safe(statement.value):
                continue
        if isinstance(statement, ast.If) and _is_main_guard(statement):
            continue
        violations.append(
            Violation(
                rule_id="script-import-side-effect",
                severity="error",
                category="mechanical",
                path=path,
                line=statement.lineno,
                column=statement.col_offset,
                message=(
                    "scripts must keep executable work inside functions and the "
                    "__main__ guard"
                ),
            )
        )


def scan_machine_specific_paths(*, path: Path, tree: ast.AST) -> list[Violation]:
    """Find machine-local absolute paths committed into Python source."""
    violations: list[Violation] = []
    if path.name == "python_standards_rules.py":
        return violations
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if not node.value.startswith(MACHINE_SPECIFIC_PATH_PREFIXES):
            continue
        violations.append(
            Violation(
                rule_id="machine-specific-path",
                severity="error",
                category="mechanical",
                path=path,
                line=node.lineno,
                column=node.col_offset,
                message="machine-specific absolute path detected",
            )
        )
    return violations


def _check_annotations(
    path: Path,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    violations: list[Violation],
) -> None:
    for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
        if argument.annotation is None:
            continue
        _check_annotation_node(
            path=path,
            annotation=argument.annotation,
            violations=violations,
            line=argument.annotation.lineno,
            column=argument.annotation.col_offset,
        )
    if node.args.vararg is not None and node.args.vararg.annotation is not None:
        _check_annotation_node(
            path=path,
            annotation=node.args.vararg.annotation,
            violations=violations,
            line=node.args.vararg.annotation.lineno,
            column=node.args.vararg.annotation.col_offset,
        )
    if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
        _check_annotation_node(
            path=path,
            annotation=node.args.kwarg.annotation,
            violations=violations,
            line=node.args.kwarg.annotation.lineno,
            column=node.args.kwarg.annotation.col_offset,
        )
    if node.returns is not None:
        _check_annotation_node(
            path=path,
            annotation=node.returns,
            violations=violations,
            line=node.returns.lineno,
            column=node.returns.col_offset,
        )


def _check_annotation_node(
    *,
    path: Path,
    annotation: ast.AST,
    violations: list[Violation],
    line: int,
    column: int,
) -> None:
    if _contains_string_annotation(annotation):
        violations.append(
            Violation(
                rule_id="forward-reference",
                severity="error",
                category="mechanical",
                path=path,
                line=line,
                column=column,
                message="forward references are forbidden",
            )
        )


def _contains_string_annotation(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            return True
    return False


def _is_module_docstring(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _is_main_guard(statement: ast.If) -> bool:
    comparison = statement.test
    if not isinstance(comparison, ast.Compare):
        return False
    if len(comparison.ops) != 1 or len(comparison.comparators) != 1:
        return False
    if not isinstance(comparison.ops[0], ast.Eq):
        return False
    left = comparison.left
    right = comparison.comparators[0]
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(right, ast.Constant)
        and right.value == "__main__"
    )


def _assignment_value_is_safe(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant | ast.Name):
        return True
    if isinstance(node, ast.Attribute):
        return _assignment_value_is_safe(node.value)
    if isinstance(node, ast.BinOp):
        return _assignment_value_is_safe(node.left) and _assignment_value_is_safe(
            node.right
        )
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, ast.UAdd | ast.USub) and _assignment_value_is_safe(
            node.operand
        )
    if isinstance(node, ast.Tuple | ast.List | ast.Set):
        return all(_assignment_value_is_safe(element) for element in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            key is None or _assignment_value_is_safe(key) for key in node.keys
        ) and all(_assignment_value_is_safe(value) for value in node.values)
    return False


def _call_target(node: ast.AST) -> tuple[str, str | None] | None:
    if isinstance(node, ast.Name):
        return (node.id, None)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return (node.value.id, node.attr)
    return None


def _render_call_target(target: tuple[str, str | None]) -> str:
    if target[1] is None:
        return target[0]
    return f"{target[0]}.{target[1]}"
