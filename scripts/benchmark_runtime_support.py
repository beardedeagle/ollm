import argparse


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_positive_int_list(value: str) -> tuple[int, ...]:
    if not value.strip():
        raise SystemExit("expected a comma-separated list of positive integers")
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise SystemExit(
            "expected a comma-separated list of positive integers"
        ) from exc
    if not values or any(item <= 0 for item in values):
        raise SystemExit("expected a comma-separated list of positive integers")
    return values
