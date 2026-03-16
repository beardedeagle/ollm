"""Check repo-local Python standards and optionally emit an audit report."""

import argparse
import ast
import asyncio
import sys
from collections import defaultdict
from pathlib import Path

from ollm.python_standards_rules import (
    StandardsVisitor,
    Violation,
    check_import_placement,
    scan_partial_work_markers,
)

DEFAULT_TARGETS: tuple[str, ...] = ("src", "tests", "scripts", "examples")
SOFT_FILE_LIMIT = 500
HARD_FILE_LIMIT = 1000


def main() -> int:
    """Run the standards checker CLI."""

    return asyncio.run(async_main())


async def async_main() -> int:
    args = parse_args()
    repo_root = args.root.expanduser().resolve()
    violations = await scan_repo(repo_root, tuple(args.target))
    rendered = render_violations(
        violations=violations,
        repo_root=repo_root,
        output_format=args.format,
    )
    if args.output is None:
        print(rendered)
    else:
        output_path = args.output.expanduser().resolve()
        await asyncio.to_thread(output_path.write_text, rendered, encoding="utf-8")
    if any(violation.severity == "error" for violation in violations):
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check repo-local Python standards and emit an audit report."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to scan.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Relative directory to scan. May be provided more than once.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "markdown"),
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output file.",
    )
    namespace = parser.parse_args()
    if not namespace.target:
        namespace.target = list(DEFAULT_TARGETS)
    return namespace


async def scan_repo(repo_root: Path, targets: tuple[str, ...]) -> list[Violation]:
    violations: list[Violation] = []
    for relative_target in targets:
        target_root = repo_root / relative_target
        if not target_root.exists():
            continue
        for path in sorted(target_root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            violations.extend(await scan_python_file(path))
    return sorted(
        violations,
        key=lambda violation: (
            violation.severity != "error",
            str(violation.path),
            violation.line,
            violation.column,
            violation.rule_id,
        ),
    )


async def scan_python_file(path: Path) -> list[Violation]:
    source = await asyncio.to_thread(path.read_text, encoding="utf-8")
    violations: list[Violation] = []
    line_count = len(source.splitlines())
    violations.extend(scan_partial_work_markers(path=path, source=source))
    if line_count > HARD_FILE_LIMIT:
        violations.append(
            Violation(
                rule_id="file-too-large-hard",
                severity="error",
                category="mechanical",
                path=path,
                line=1,
                column=0,
                message=(
                    f"file has {line_count} lines and exceeds the hard limit of "
                    f"{HARD_FILE_LIMIT}"
                ),
            )
        )
    elif line_count > SOFT_FILE_LIMIT:
        violations.append(
            Violation(
                rule_id="file-too-large-soft",
                severity="audit",
                category="structural",
                path=path,
                line=1,
                column=0,
                message=(
                    f"file has {line_count} lines and exceeds the soft limit of "
                    f"{SOFT_FILE_LIMIT}"
                ),
            )
        )
    tree = ast.parse(source, filename=str(path))
    check_import_placement(path=path, tree=tree, violations=violations)
    visitor = StandardsVisitor(path)
    visitor.visit(tree)
    violations.extend(visitor.violations)
    return violations


def render_violations(
    *, violations: list[Violation], repo_root: Path, output_format: str
) -> str:
    if output_format == "markdown":
        return render_markdown(violations=violations, repo_root=repo_root)
    return render_text(violations=violations, repo_root=repo_root)


def render_text(*, violations: list[Violation], repo_root: Path) -> str:
    if not violations:
        return "Python standards check passed with no violations."
    grouped = _group_by_rule(violations)
    lines = [
        "Python standards report",
        "=======================",
        "",
        f"Total violations: {len(violations)}",
        f"Error violations: {sum(1 for violation in violations if violation.severity == 'error')}",
        f"Audit-only findings: {sum(1 for violation in violations if violation.severity == 'audit')}",
        "",
    ]
    for rule_id in sorted(grouped):
        lines.append(f"[{rule_id}]")
        for violation in grouped[rule_id]:
            relative_path = violation.path.relative_to(repo_root)
            lines.append(
                (
                    f"- {relative_path}:{violation.line}:{violation.column + 1} "
                    f"[{violation.severity}] {violation.message}"
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def render_markdown(*, violations: list[Violation], repo_root: Path) -> str:
    mechanical = [
        violation for violation in violations if violation.category == "mechanical"
    ]
    other = [
        violation for violation in violations if violation.category != "mechanical"
    ]
    lines = [
        "# Python Standards Audit",
        "",
        "This report is generated from `scripts/check_python_standards.py` against the current branch.",
        "",
        f"- Total findings: {len(violations)}",
        f"- Mechanical failures: {sum(1 for violation in violations if violation.severity == 'error')}",
        f"- Audit-only findings: {sum(1 for violation in violations if violation.severity == 'audit')}",
        "",
        "## Mechanical failures",
        "",
    ]
    if mechanical:
        lines.extend(_render_markdown_table(mechanical, repo_root))
    else:
        lines.append("No mechanical failures remain.")
    lines.extend(["", "## Audit findings", ""])
    if other:
        lines.extend(_render_markdown_table(other, repo_root))
    else:
        lines.append("No audit-only findings remain.")
    return "\n".join(lines).rstrip()


def _render_markdown_table(violations: list[Violation], repo_root: Path) -> list[str]:
    lines = [
        "| Rule | Severity | File | Location | Message |",
        "| --- | --- | --- | --- | --- |",
    ]
    for violation in violations:
        relative_path = violation.path.relative_to(repo_root)
        lines.append(
            (
                f"| `{violation.rule_id}` | `{violation.severity}` | "
                f"`{relative_path}` | `{violation.line}:{violation.column + 1}` | "
                f"{violation.message} |"
            )
        )
    return lines


def _group_by_rule(violations: list[Violation]) -> dict[str, list[Violation]]:
    grouped: dict[str, list[Violation]] = defaultdict(list)
    for violation in violations:
        grouped[violation.rule_id].append(violation)
    return dict(grouped)


if __name__ == "__main__":
    sys.exit(main())
