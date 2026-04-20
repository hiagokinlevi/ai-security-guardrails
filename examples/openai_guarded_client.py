"""Example: protected OpenAI client usage with ai-security-guardrails.

This example demonstrates a minimal pattern for:
1) Loading guardrail configuration
2) Validating/screening user input before model invocation
3) Filtering/screening model output before returning to user

Environment variables:
- OPENAI_API_KEY: required
- OPENAI_MODEL: optional (default: gpt-4o-mini)
- GUARDRAILS_CONFIG: optional path to YAML policy/config

Run:
    python examples/openai_guarded_client.py
"""

from __future__ import annotations

import os
import sys

from openai import OpenAI

# The project exposes guardrail primitives and config loading.
# Import paths are intentionally local to this repository.
from guardrails.config import load_config
from guardrails.wrapper import GuardrailsWrapper


def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set", file=sys.stderr)
        return 1

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    config_path = os.getenv("GUARDRAILS_CONFIG")

    # Load guardrails configuration (defaults if path is omitted).
    config = load_config(config_path) if config_path else load_config()
    guardrails = GuardrailsWrapper(config=config)

    client = OpenAI(api_key=api_key)

    user_prompt = "Provide a brief checklist for securing an internal API gateway."

    # 1) Pre-flight input checks
    input_result = guardrails.validate_input(user_prompt)
    if getattr(input_result, "blocked", False):
        print("Input blocked by guardrails:", getattr(input_result, "reason", "policy violation"))
        return 2

    prompt_for_model = getattr(input_result, "sanitized_text", None) or user_prompt

    # 2) Model invocation
    response = client.responses.create(
        model=model,
        input=prompt_for_model,
    )

    raw_output = getattr(response, "output_text", "")

    # 3) Post-flight output checks
    output_result = guardrails.validate_output(raw_output)
    if getattr(output_result, "blocked", False):
        print("Output blocked by guardrails:", getattr(output_result, "reason", "policy violation"))
        return 3

    safe_output = getattr(output_result, "sanitized_text", None) or raw_output
    print("\nGuarded model response:\n")
    print(safe_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
