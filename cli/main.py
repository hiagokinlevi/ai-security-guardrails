from __future__ import annotations

import argparse
import sys

from guardrails.audit import verify_audit_chain_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="guardrails")
    subparsers = parser.add_subparsers(dest="command")

    verify_parser = subparsers.add_parser(
        "verify-audit-log",
        help="Verify newline-delimited audit log hash-chain integrity",
    )
    verify_parser.add_argument("path", help="Path to NDJSON audit log")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "verify-audit-log":
        ok, bad_index, reason = verify_audit_chain_file(args.path)
        if ok:
            print("audit log integrity: OK")
            return 0
        print(f"audit log integrity: FAIL at event index {bad_index} ({reason})")
        return 1

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
