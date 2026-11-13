"""Read-only, fixed-command DVC/Git adapter. Historical checkout is an explicit documented workflow."""

import json
import subprocess

import yaml

from agent.config import Settings


class DVCTools:
    def __init__(self, settings: Settings):
        self.settings = settings

    def command(self, *args: str) -> str:
        process = subprocess.run(
            args,
            cwd=self.settings.repo,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.settings.tool_timeout,
            check=False,
        )
        if process.returncode:
            raise RuntimeError((process.stderr or process.stdout)[-2000:])
        return process.stdout.strip()

    def revision(self, revision: str) -> str:
        # Resolve to an immutable hash before interpolating into Git's object:path syntax.
        return self.command("git", "rev-parse", "--verify", "--end-of-options", revision + "^{commit}")

    def list_datasets(self) -> dict:
        import sys

        return {
            "datasets": json.loads(
                self.command(sys.executable, "-m", "dvc", "list", ".", "--dvc-only", "--recursive", "--json")
            )
        }

    def dataset_version(self, revision: str = "HEAD") -> dict:
        commit = self.revision(revision)
        metadata = yaml.safe_load(self.command("git", "show", f"{commit}:data/iris.csv.dvc"))
        return {
            "git_commit": commit,
            "reference": revision,
            "dataset": "data/iris.csv",
            "outputs": metadata["outs"],
        }

    def compare_versions(self, old_revision: str, new_revision: str = "HEAD") -> dict:
        import sys

        old, new = self.revision(old_revision), self.revision(new_revision)
        result = json.loads(self.command(sys.executable, "-m", "dvc", "diff", old, new, "--json"))
        return {"old_commit": old, "new_commit": new, "diff": result}

    def status(self) -> dict:
        import sys

        return {"status": json.loads(self.command(sys.executable, "-m", "dvc", "data", "status", "--json"))}

    def reproduction(self, revision: str = "HEAD") -> dict:
        result = self.dataset_version(revision)
        commit = result["git_commit"]
        return result | {
            "executed": False,
            "instructions": [
                f"git worktree add --detach ../dataset-reproduction {commit}",
                "cd ../dataset-reproduction",
                "dvc pull data/iris.csv.dvc",
                "dvc checkout data/iris.csv.dvc",
                "dvc data status --json",
            ],
            "prerequisite": (
                "DVC remote must contain this hash. Run scripts/bootstrap_data.py for demo data."
            ),
        }
