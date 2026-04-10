# Overview

## What is ai-security-guardrails?

`ai-security-guardrails` is a Python library that provides a policy-driven security layer for
applications built on large language models (LLMs). It is designed to be inserted between the
application's user-facing interface and the underlying model API.

## Who is it for?

The library is intended for:

- **Application developers** building production LLM applications (chatbots, copilots, agents, RAG
  systems) who want to add security controls without building them from scratch.
- **Security engineers** responsible for reviewing and auditing AI applications.
- **Platform teams** who want to enforce consistent security policies across multiple LLM-based
  services.

## Core concepts

### Risk scoring

Every user input and every model output receives a numeric risk score between 0.0 (clean) and 1.0
(high risk). The score is computed by applying a set of heuristic rules (pattern matching, length
checks, etc.) and summing the weights of any matched rules.

Risk scores are **probabilistic signals**, not ground truth. A score of 0.8 does not mean the
content is definitely malicious — it means it matched enough heuristic patterns to warrant elevated
scrutiny. The policy engine translates scores into actions.

### Policy engine

The policy engine loads a YAML configuration file and maps risk scores and signal types to actions:

- `allow` — proceed without restriction
- `allow_with_warning` — proceed but flag in the audit log
- `send_to_review` — queue for human review
- `block` — reject and return a safe error message

### Audit trail

Every guardrail decision is recorded as a structured JSON event. The audit trail is the primary
mechanism for governance, compliance review, and incident investigation.

### Offline validation CLI

`k1n-guardrails validate-input` and `k1n-guardrails detect-injection` provide JSON output for CI
jobs, pre-deployment review, and incident-response triage where teams need to evaluate a prompt,
retrieved context file, or tool result without calling an external model API.

The input validator scans both the raw text and a normalized view that collapses common leetspeak,
zero-width characters, punctuation, and spacing tricks used to disguise instruction-override
phrases. Review and block decisions return nonzero exit codes so shell pipelines can fail closed.

For tenant or application-specific controls, pass `--regex-rule-set` or call `validate_input()` with
`regex_rule_set_path`. YAML-backed rules are compiled locally, stay offline, and add reviewed flags
and scores to the same validation result.

## What it is not

- **Not a firewall** — it does not inspect network traffic or enforce network-level policies.
- **Not a model fine-tuning tool** — it operates at the application layer, not the model layer.
- **Not a complete security solution** — it is one layer in a defense-in-depth strategy.
  Application authentication, authorization, and network security are outside its scope.
