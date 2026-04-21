#!/usr/bin/env python3
"""Simple CLI to scan a prompt for prompt-injection risk."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure local project imports work when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guardrails.input_guard import PromptInjectionDetector  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan a prompt for prompt-injection patterns and risk score."
    )
    parser.add_argument("prompt", help="Prompt text to scan")
    args = parser.parse_args()

    detector = PromptInjectionDetector()
    result = detector.detect(args.prompt)

    output = {
        "risk_score": result.risk_score,
        "flagged_patterns": result.flagged_patterns,
    }
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
