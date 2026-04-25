from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from guardrails.policy import Policy


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Policy file must contain a YAML mapping/object at the top level")
    return data


def cmd_validate_policy(policy_path: str) -> int:
    path = Path(policy_path)
    if not path.exists():
        print(f"❌ Policy file not found: {path}", file=sys.stderr)
        return 2

    try:
        raw = _load_yaml(path)
        Policy.model_validate(raw)
    except (yaml.YAMLError, ValueError) as e:
        print(f"❌ Invalid policy YAML: {e}", file=sys.stderr)
        return 2
    except ValidationError as e:
        print("❌ Policy schema validation failed:", file=sys.stderr)
        print(e, file=sys.stderr)
        return 1

    print(f"✅ Policy is valid: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-guardrails")
    subparsers = parser.add_subparsers(dest="command")

    validate_parser = subparsers.add_parser(
        "validate-policy",
        help="Validate a policy YAML file against the policy schema",
    )
    validate_parser.add_argument("--policy", required=True, help="Path to policy YAML file")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-policy":
        return cmd_validate_policy(args.policy)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
