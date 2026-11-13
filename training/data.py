"""Generate deterministic, redistributable versions of scikit-learn's bundled Iris data."""

import argparse
import csv
from pathlib import Path

from sklearn.datasets import load_iris

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def export_dataset(version: str, destination: Path) -> None:
    iris = load_iris()
    # v1 contains the first 40 samples of each class; v2 includes all 50.
    per_class = {"v1": 40, "v2": 50}[version]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([*FEATURES, "target"])
        for index, (row, target) in enumerate(zip(iris.data, iris.target, strict=True)):
            if index % 50 < per_class:
                writer.writerow([*row, int(target)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--output", type=Path, default=Path("data/iris.csv"))
    args = parser.parse_args()
    export_dataset(args.version, args.output)
