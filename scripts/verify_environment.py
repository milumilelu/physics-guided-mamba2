"""Fail fast when the active interpreter differs from the tested environment."""

from __future__ import annotations

import importlib
import sys


EXPECTED_PYTHON = (3, 12, 13)
EXPECTED_PACKAGES = {
    "numpy": "2.3.5",
    "pandas": "3.0.1",
    "PIL": "12.3.0",
    "matplotlib": "3.10.7",
    "sklearn": "1.7.2",
    "scipy": "1.18.1",
}
OPTIONAL_PACKAGES = {
    "catboost": "1.2.10",
    "xgboost": "3.4.1",
}


def main() -> int:
    """Check exact interpreter/package versions and report optional CUDA state."""
    failures: list[str] = []
    actual_python = sys.version_info[:3]
    if actual_python != EXPECTED_PYTHON:
        failures.append(
            f"Python mismatch: expected {EXPECTED_PYTHON}, got {actual_python}"
        )

    for module_name, expected in EXPECTED_PACKAGES.items():
        module = importlib.import_module(module_name)
        actual = str(module.__version__)
        print(f"{module_name}={actual}")
        if actual != expected:
            failures.append(
                f"{module_name} mismatch: expected {expected}, got {actual}"
            )

    for module_name, expected in OPTIONAL_PACKAGES.items():
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            print(f"{module_name}=not installed (optional model extension)")
            continue
        actual = str(module.__version__)
        print(f"{module_name}={actual}")
        if actual != expected:
            failures.append(
                f"{module_name} mismatch: expected {expected}, got {actual}"
            )

    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        print("torch=not installed (optional CUDA extension)")
    else:
        print(f"torch={torch.__version__}")
        print(f"cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda_device={torch.cuda.get_device_name(0)}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    print("environment=OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
