"""Deprecated shim: use scripts/02_build_topic_assignment.py."""

from __future__ import annotations

import runpy
import sys
import warnings
from pathlib import Path

warnings.warn(
    "02_build_topic_entropy.py is deprecated; use 02_build_topic_assignment.py",
    DeprecationWarning,
    stacklevel=1,
)
target = Path(__file__).resolve().parent / "02_build_topic_assignment.py"
sys.argv[0] = str(target)
runpy.run_path(str(target), run_name="__main__")
