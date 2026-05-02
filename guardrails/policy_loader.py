from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeConfig:
    policy_file_path: str
    policy_file_sha256: str


def load_policy(policy_path: str | Path) -> tuple[dict[str, Any], RuntimeConfig]:
    path = Path(policy_path)
    raw_bytes = path.read_bytes()
    policy = yaml.safe_load(raw_bytes) or {}

    digest = hashlib.sha256(raw_bytes).hexdigest()
    runtime_config = RuntimeConfig(
        policy_file_path=str(path),
        policy_file_sha256=digest,
    )

    logger.info(
        "Policy initialized",
        extra={
            "policy_file_path": runtime_config.policy_file_path,
            "policy_file_sha256": runtime_config.policy_file_sha256,
        },
    )

    return policy, runtime_config
