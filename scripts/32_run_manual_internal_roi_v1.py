#!/usr/bin/env python3
"""One-command entry for the fast manual stable-ROI extraction."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    commands = [
        [sys.executable, str(REPO/"scripts/verify_environment.py")],
        [sys.executable, "-m", "unittest", "discover", "-s",
         str(REPO/"tests"), "-v"],
        [sys.executable, str(REPO/"scripts/22_extract_stable_roi_fast.py")],
        [sys.executable, str(REPO/"scripts/23_build_stable_roi_dataset.py")],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=REPO)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
