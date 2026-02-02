#!/usr/bin/env python3
"""
Runner for SEE ensemble experiments. Use with uv from project root:

    uv sync
    uv run python run_see.py --datasets albrecht kemerer [--max-iter 5]
"""

import sys
from pathlib import Path

# Ensure code/ is on path so ensemble_see is importable
_root = Path(__file__).resolve().parent
_code = _root / "code"
if _code.exists() and str(_code) not in sys.path:
    sys.path.insert(0, str(_code))

from ensemble_see.run import main

if __name__ == "__main__":
    sys.exit(main())
