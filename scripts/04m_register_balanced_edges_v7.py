#!/usr/bin/env python3
"""Run the generic v6/v7 runner with the frozen v7 configuration."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("04i_register_step_consensus_v6.py")
sys.argv = [str(SCRIPT), "--version", "v7"]
runpy.run_path(str(SCRIPT), run_name="__main__")
