"""Rebuild both public demo versions in DVC's cache and local remote after a fresh clone."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.data import export_dataset  # noqa: E402


def dvc(*arguments):
    subprocess.run([sys.executable, "-m", "dvc", *arguments], cwd=ROOT, check=True)


def main():
    pointer = ROOT / "data/iris.csv.dvc"
    original = pointer.read_bytes()
    data = ROOT / "data/iris.csv"
    previous_data = data.read_bytes() if data.exists() else None
    succeeded = False
    try:
        for version in ("v1", "v2"):
            export_dataset(version, data)
            dvc("add", "data/iris.csv")
            dvc("push", "data/iris.csv.dvc")
        succeeded = True
    finally:
        pointer.write_bytes(original)
        if previous_data is not None:
            data.write_bytes(previous_data)
        if succeeded:
            dvc("checkout", "data/iris.csv.dvc")
    print("Both dataset versions cached and pushed; original committed reference restored.")


if __name__ == "__main__":
    main()
