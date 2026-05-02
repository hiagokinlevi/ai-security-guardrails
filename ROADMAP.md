# Roadmap

This document outlines the planned development trajectory for `ai-security-guardrails`.
Priorities may shift based on community feedback and emerging threat landscape.

---

## v0.1 — Foundation (Current)

- [x] Input validation with heuristic risk scoring
- [x] Output filtering and PII redaction
- [x] YAML-based policy engine (allow / warn / block)
- [x] Structured audit logging
- [x] FastAPI middleware integration
- [x] Default policy for general-purpose LLM apps
- [x] Pydantic schemas for all events
- [x] Offline JSON CLI for input validation and prompt injection scans
- [x] Basic obfuscation normalization for instruction-override detection

---

## v0.2 — Improved Detection

- [ ] Embedding-based injection detection (semantic similarity to known injection patterns)
- [x] Configurable regex rule sets loaded from YAML
- [ ] Per-tenant policy support (multi-tenant applications)
- [x] Token budget enforcement via tiktoken integration
- [ ] Rate limiting per user / session

---

## v0.3 — Agent and Tool Security

- [ ] Tool call interceptor for OpenAI function calling
- [ ] Tool allow-list and deny-list enforcement with structured reasons
- [x] Maximum tool call depth per agent turn
- [ ] Sandboxed tool execution result validation
- [ ] Agent loop detection (infinite tool call cycles)

---

## v0.4 — RAG Security

- [ ] Retrieved context scanning before injection into prompts
- [ ] Indirect prompt injection detection in retrieved documents
- [ ] Source attribution enforcement (require citations for factual claims)
- [ ] Chunked context risk scoring

---

## v0.5 — Observability and Governance

- [ ] OpenTelemetry trace integration
- [ ] Prometheus metrics exporter (policy decisions, block rates, latency)
- [ ] Grafana dashboard template
- [ ] Policy version ma

## Automated Completions
- [x] Add deterministic policy_decision_reason_code enum with schema validation (cycle 43)
