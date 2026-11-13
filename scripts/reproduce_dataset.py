"""Verify a historical dataset in a detached temporary worktree without altering working data."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default="dataset-v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", args.revision + "^{commit}"], cwd=ROOT, text=True
    ).strip()
    # Resolve the repository's documented demo remote; no user credentials or remote URLs are exported.
    remote = str((ROOT / ".runtime/dvc-remote").resolve())
    with tempfile.TemporaryDirectory(prefix="mlops-dataset-") as directory:
        worktree = Path(directory) / "checkout"
        subprocess.run(["git", "worktree", "add", "--detach", str(worktree), commit], cwd=ROOT, check=True)
        try:

            def dvc(*values):
                return subprocess.check_output(
                    [sys.executable, "-m", "dvc", *values], cwd=worktree, text=True
                )

            # Reuse the absolute local remote in this separate checkout.
            dvc("remote", "modify", "--local", "local", "url", remote)
            dvc("pull", "data/iris.csv.dvc")
            dvc("checkout", "data/iris.csv.dvc")
            expected = yaml.safe_load((worktree / "data/iris.csv.dvc").read_text())["outs"][0]["md5"]
            actual = hashlib.md5((worktree / "data/iris.csv").read_bytes()).hexdigest()
            if expected != actual:
                raise RuntimeError("Restored dataset digest differs from recorded DVC reference")
            result = {
                "revision": args.revision,
                "git_commit": commit,
                "expected_dvc_hash": expected,
                "restored_hash": actual,
                "matches": True,
                "rows": len((worktree / "data/iris.csv").read_text().splitlines()) - 1,
                "dvc_status": json.loads(dvc("data", "status", "--json")),
            }
            print(json.dumps(result, indent=2))
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
