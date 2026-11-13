"""Create local-only credentials without printing them or overwriting existing configuration."""

import argparse
import json
import secrets
import sys
from pathlib import Path

from cryptography.fernet import Fernet
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-env", type=Path, help="Optional existing .env with OPENROUTER_API_KEY")
    args = parser.parse_args()
    target = ROOT / ".env"
    existing = dotenv_values(target) if target.exists() else {}
    values = dict(dotenv_values(ROOT / ".env.example")) | existing
    if args.reuse_env and not values.get("OPENROUTER_API_KEY"):
        previous = dotenv_values(args.reuse_env)
        values["OPENROUTER_API_KEY"] = previous.get("OPENROUTER_API_KEY", "")
    defaults = {
        "AIRFLOW_PASSWORD": secrets.token_urlsafe(24),
        "POSTGRES_PASSWORD": secrets.token_hex(24),
        "AIRFLOW_JWT_SECRET": secrets.token_hex(32),
        "AIRFLOW_FERNET_KEY": Fernet.generate_key().decode(),
    }
    for key, value in defaults.items():
        if not values.get(key):
            values[key] = value
    target.write_text("".join(f"{key}={value or ''}\n" for key, value in values.items()), encoding="utf-8")
    secret_dir = ROOT / ".runtime/airflow-secrets"
    secret_dir.mkdir(parents=True, exist_ok=True)
    (secret_dir / "passwords.json").write_text(json.dumps({"admin": values["AIRFLOW_PASSWORD"]}))
    (ROOT / ".runtime/training").mkdir(parents=True, exist_ok=True)
    print("Local .env and Airflow credentials configured; secret values were not printed.")
    if not values.get("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY in .env before using the LLM agent.", file=sys.stderr)


if __name__ == "__main__":
    main()
