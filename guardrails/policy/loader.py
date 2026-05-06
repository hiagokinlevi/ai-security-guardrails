from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class PolicyDecisionReasonCodeInventory:
    reason_codes: tuple[str, ...]
    sha256: str


def build_policy_decision_reason_code_inventory(
    allowed_reason_codes: Iterable[str],
) -> PolicyDecisionReasonCodeInventory:
    """Build deterministic inventory and digest for policy_decision_reason_code values.

    Inventory is sorted and deduplicated to provide a stable, tamper-detectable baseline
    across deploys.
    """
    normalized = tuple(sorted({code.strip() for code in allowed_reason_codes if code and code.strip()}))
    serialized = "\n".join(normalized).encode("utf-8")
    digest = hashlib.sha256(serialized).hexdigest()
    return PolicyDecisionReasonCodeInventory(reason_codes=normalized, sha256=digest)
