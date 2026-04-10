"""Command-line workflows for AI Security Guardrails."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from guardrails.input_controls.validator import InputDecision, validate_input
from guardrails.prompt_injection.detector import detect_injection


def _read_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.file is not None:
        with open(args.file, encoding="utf-8") as handle:
            return handle.read()
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("Provide --text, --file, or stdin input.")


def _validate_input(args: argparse.Namespace) -> int:
    result = validate_input(
        _read_text(args),
        max_length=args.max_length,
        risk_threshold=args.risk_threshold,
        regex_rule_set_path=args.regex_rule_set,
    )
    payload = {
        "decision": result.decision.value,
        "risk_score": result.risk_score,
        "risk_flags": sorted(result.risk_flags),
        "reason": result.reason,
        "original_length": result.original_length,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if result.decision in (InputDecision.SEND_TO_REVIEW, InputDecision.BLOCK) else 0


def _detect_injection(args: argparse.Namespace) -> int:
    report = detect_injection(
        _read_text(args),
        source_type=args.source_type,
        include_template_checks=not args.no_template_checks,
    )
    payload = {
        "risk_level": report.risk_level.value,
        "source_type": report.source_type,
        "signals": [
            {
                "rule_id": signal.rule_id,
                "confidence": signal.confidence.value,
                "category": signal.category,
                "description": signal.description,
                "matched_text": signal.matched_text,
            }
            for signal in report.signals
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if report.has_signals else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="k1n-guardrails",
        description="Offline AI guardrail validation workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate-input",
        help="Risk-score user input before sending it to an LLM.",
    )
    validate.add_argument("--text", help="Text to validate.")
    validate.add_argument("--file", help="UTF-8 text file to validate.")
    validate.add_argument("--max-length", type=int, default=10000)
    validate.add_argument("--risk-threshold", type=float, default=0.7)
    validate.add_argument(
        "--regex-rule-set",
        help="Optional YAML file with application-specific regex rules.",
    )
    validate.set_defaults(func=_validate_input)

    injection = subparsers.add_parser(
        "detect-injection",
        help="Scan direct input, retrieved context, or tool output for injection signals.",
    )
    injection.add_argument("--text", help="Text to scan.")
    injection.add_argument("--file", help="UTF-8 text file to scan.")
    injection.add_argument(
        "--source-type",
        choices=["direct", "indirect", "tool_output"],
        default="direct",
    )
    injection.add_argument("--no-template-checks", action="store_true")
    injection.set_defaults(func=_detect_injection)

    return parser


def cli(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def main() -> None:
    raise SystemExit(cli())


if __name__ == "__main__":
    main()
