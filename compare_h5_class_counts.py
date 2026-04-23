"""Convenience wrapper.

Allows running:
  python -u compare_h5_class_counts.py --a ... --b ...

which delegates to scripts/compare_h5_class_counts.py.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
  try:
    import h5py  # noqa: F401
    import numpy  # noqa: F401
  except ModuleNotFoundError as e:
    missing = getattr(e, "name", "<unknown>")
    raise SystemExit(
      "Missing Python dependency: "
      + str(missing)
      + "\n\n"
      + "Run this under the conda env that has h5py installed (e.g. fer_augmentation), "
      + "or submit the provided sbatch job.\n\n"
      + "Example (interactive on cluster):\n"
      + "  module load miniconda3/3.13.25\n"
      + "  source \"$(conda info --base)/etc/profile.d/conda.sh\"\n"
      + "  conda activate fer_augmentation\n"
      + "  python -u compare_h5_class_counts.py --a ~/data/dataset.h5 --b ~/data/dataset_augmented.h5\n\n"
      + "Example (batch):\n"
      + "  sbatch sbatch/compare_h5_class_counts.sbatch\n"
    )

    script_path = Path(__file__).resolve().parent / "scripts" / "compare_h5_class_counts.py"
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
