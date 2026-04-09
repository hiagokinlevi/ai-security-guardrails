# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.x     | Yes       |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability in `ai-security-guardrails`, please report it
responsibly by emailing the maintainer directly. Include:

1. A description of the vulnerability and its potential impact.
2. Steps to reproduce the issue.
3. Any proof-of-concept code (if applicable).
4. Your suggested fix or mitigation (optional but appreciated).

You can expect an acknowledgment within **48 hours** and a resolution plan within **7 days**
for confirmed critical issues.

## Scope

The following are in scope for security reports:

- Bypasses of the input validation or output filtering logic
- Vulnerabilities in the policy engine that allow policy evasion
- Issues that could cause sensitive data to be logged or leaked via the audit trail
- Dependency vulnerabilities with a direct exploit path

The following are **out of scope**:

- Attacks that require full control of the host system
- Issues in upstream dependencies without a direct exploit path in this library
- Social engineering attacks

## Security Design Principles

This library is built around the following security principles:

1. **Fail closed** — When in doubt, block and log. A false positive is preferable to a data breach.
2. **Defense in depth** — Input controls, output controls, and audit logging are independent layers.
   Defeating one layer does not defeat all layers.
3. **No raw data in logs** — Audit logs record decisions, scores, and flags — not raw inputs or
   outputs. Raw content must be explicitly enabled and should only be used in isolated debug
   environments.
4. **Explicit allow-lists for tools** — Agent tools are blocked by default. Each allowed tool must
   be explicitly listed in the policy.
5. **Heuristic, not deterministic** — Risk scores are probabilistic signals. They should inform
   human review, not replace it for high-stakes decisions.

## Disclosure Policy

We follow a **coordinated disclosure** model. Once a vulnerability is confirmed and a fix is
available, we will:

1. Publish a security advisory on GitHub.
2. Release a patched version.
3. Credit the reporter (unless they prefer anonymity).
