from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import yaml


ALLOWED_POLICY_MODES = {0o600, 0o640}
POLICY_PERMISSIONS_OVERRIDE_ENV = "AIGUARD_ALLOW_INSECURE_POLICY_PERMISSIONS"


class PolicySecurityError(RuntimeError):
    """Raised when policy file security validation fails."""


def _format_mode(mode: int) -> str:
    return f"0o{mode:03o}"


def _is_override_enabled() -> bool:
    value = os.getenv(POLICY_PERMISSIONS_OVERRIDE_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def validate_policy_file_permissions(policy_path: str | Path) -> None:
    """
    Validate policy file mode is secure by default.

    Allowed modes:
      - 0o600
      - 0o640

    Set AIGUARD_ALLOW_INSECURE_POLICY_PERMISSIONS=1 to bypass this check
    for controlled containerized deployments.
    """
    path = Path(policy_path)
    st = path.stat()
    mode = stat.S_IMODE(st.st_mode)

    if mode in ALLOWED_POLICY_MODES:
        return

    if _is_override_enabled():
        return

    allowed = ", ".join(_format_mode(m) for m in sorted(ALLOWED_POLICY_MODES))
    current = _format_mode(mode)
    raise PolicySecurityError(
        "Insecure policy file permissions detected for "
        f"'{path}'. Current mode is {current}; allowed modes are {allowed}. "
        "Fix with: chmod 600 <policy-file> (or 640 when group-read is required). "
        f"For controlled exceptions (e.g., containerized deployments), set "
        f"{POLICY_PERMISSIONS_OVERRIDE_ENV}=1."
    )


def load_policy(policy_path: str | Path) -> dict[str, Any]:
    """Load policy YAML after startup security checks."""
    validate_policy_file_permissions(policy_path)
    with Path(policy_path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Policy file must contain a top-level YAML mapping")
    return data
