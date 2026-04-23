"""Convenience wrapper.

Allows running:
  python -u compare_h5_class_counts.py --a ... --b ...

which delegates to scripts/compare_h5_class_counts.py.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    script_path = Path(__file__).resolve().parent / "scripts" / "compare_h5_class_counts.py"
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
