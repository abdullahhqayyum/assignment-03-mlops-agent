"""Redact known secrets, credential assignments and token patterns before logs or responses."""

import json
import logging
import re
from typing import Any

SENSITIVE = re.compile(r"password|secret|token|api[_-]?key|authorization|credential", re.I)
TOKEN = re.compile(
    r"\b(?:sk-or-v1-[a-zA-Z0-9-]+|sk-[a-zA-Z0-9_-]{16,}|gh[pousr]_[a-zA-Z0-9_]+|eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)"
)
ASSIGNMENT = re.compile(r"(?i)((?:password|secret|token|api[_-]?key|authorization)\s*[:=]\s*)[^\s,;]+")


class Redactor:
    def __init__(self, secrets: list[str] | None = None):
        self.secrets = [value for value in secrets or [] if value]

    def clean(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(k): "[REDACTED]" if SENSITIVE.search(str(k)) else self.clean(v) for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.clean(v) for v in value]
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, "[REDACTED]")
            value = re.sub(r"(?i)\bbearer\s+[A-Za-z0-9_.~+/-]+=*", "Bearer [REDACTED]", value)
            value = re.sub(r"(?<=://)[^/@\s]+:[^/@\s]+@", "[REDACTED]@", value)
            value = TOKEN.sub("[REDACTED]", value)
            return ASSIGNMENT.sub(r"\1[REDACTED]", value)
        return value


def log_event(event: dict) -> None:
    logging.getLogger("mlops.tools").info(json.dumps(event, allow_nan=False))
