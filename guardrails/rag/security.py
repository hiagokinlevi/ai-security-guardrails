"""
RAG Pipeline Security
======================
Security controls for Retrieval-Augmented Generation (RAG) pipelines.

When an LLM retrieves documents from a vector store, knowledge base, or
external source, those documents may contain attacker-controlled content
designed to hijack the model's behaviour (indirect prompt injection).

This module provides:
  1. Document source validation — ensure retrieved documents come from
     approved sources (allowlisted URLs, namespaces, or metadata tags)
  2. Context poisoning detection — scan retrieved chunks for embedded
     injection instructions before they reach the LLM context window
  3. Context window size enforcement — prevent context stuffing attacks

Usage:
    from guardrails.rag.security import RagSecurityConfig, validate_rag_context

    config = RagSecurityConfig(
        allowed_sources={"internal-kb", "docs.mycompany.com"},
        max_total_chars=16000,
    )
    result = validate_rag_context(documents, config)
    if not result.passed:
        # Remove or quarantine flagged documents before building the prompt
        safe_docs = [d for d in documents if d not in result.flagged_documents]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from guardrails.prompt_injection.detector import (
    InjectionReport,
    RiskLevel,
    detect_injection,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RagSecurityConfig:
    """
    Configuration for RAG pipeline security checks.

    Args:
        allowed_sources:        Set of approved source identifiers (URLs,
                                namespace names, or metadata tags). If None,
                                source validation is skipped.
        max_total_chars:        Maximum total characters across all retrieved
                                documents. Prevents context-stuffing attacks
                                that push the LLM context window to its limit.
        max_single_doc_chars:   Maximum characters per individual document chunk.
        min_injection_risk:     Minimum risk level that causes a document to be
                                flagged. Default: MEDIUM (flag MEDIUM, HIGH, CRITICAL).
        quarantine_on_flag:     If True, flagged documents are excluded from the
                                safe document list. If False, they are only warned.
    """
    allowed_sources: Optional[set[str]] = None
    max_total_chars: int = 32_000       # ~8k tokens at 4 chars/token
    max_single_doc_chars: int = 8_000   # ~2k tokens per chunk
    min_injection_risk: RiskLevel = RiskLevel.MEDIUM
    quarantine_on_flag: bool = True


# ---------------------------------------------------------------------------
# Document model
# ---------------------------------------------------------------------------

@dataclass
class RagDocument:
    """
    A document chunk retrieved from the knowledge base.

    Args:
        content:   The text content of the document chunk.
        source:    Source identifier (URL, filename, namespace, doc ID).
        metadata:  Optional dict of additional metadata for context.
    """
    content: str
    source: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class DocumentCheckResult:
    """Security check result for a single document."""

    document: RagDocument
    passed: bool
    issues: list[str]                       # Human-readable issue descriptions
    injection_report: Optional[InjectionReport] = None


@dataclass
class RagSecurityResult:
    """Aggregated result from validating a full RAG context."""

    total_documents: int
    flagged_documents: list[RagDocument]
    safe_documents: list[RagDocument]
    doc_results: list[DocumentCheckResult]
    total_chars: int
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if no documents were flagged for quarantine."""
        return len(self.flagged_documents) == 0

    @property
    def flag_count(self) -> int:
        return len(self.flagged_documents)


# ---------------------------------------------------------------------------
# Risk level ordering for comparison
# ---------------------------------------------------------------------------

_RISK_ORDER = {
    RiskLevel.CLEAN: 0,
    RiskLevel.LOW: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH: 3,
    RiskLevel.CRITICAL: 4,
}


def _risk_meets_threshold(risk: RiskLevel, threshold: RiskLevel) -> bool:
    """Return True if risk is at or above the threshold level."""
    return _RISK_ORDER.get(risk, 0) >= _RISK_ORDER.get(threshold, 0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_rag_context(
    documents: list[RagDocument],
    config: Optional[RagSecurityConfig] = None,
) -> RagSecurityResult:
    """
    Validate a list of retrieved RAG documents for security issues.

    Performs three checks per document:
      1. Source allowlist — document source must be in allowed_sources (if configured)
      2. Size limit — individual document may not exceed max_single_doc_chars
      3. Injection scan — document content is scanned for embedded injection patterns

    Additionally checks the total context window size across all documents.

    Args:
        documents: List of retrieved document chunks.
        config:    Security configuration. If None, uses conservative defaults.

    Returns:
        RagSecurityResult with flagged/safe document lists and per-doc results.
    """
    cfg = config or RagSecurityConfig()
    doc_results: list[DocumentCheckResult] = []
    flagged: list[RagDocument] = []
    safe: list[RagDocument] = []
    warnings: list[str] = []
    total_chars = sum(len(d.content) for d in documents)

    if total_chars > cfg.max_total_chars:
        warnings.append(
            f"Total RAG context size ({total_chars} chars) exceeds limit "
            f"({cfg.max_total_chars} chars). Truncate or filter documents before "
            "building the prompt to prevent context-stuffing."
        )

    for doc in documents:
        issues: list[str] = []
        injection_report: Optional[InjectionReport] = None

        # --- Source validation ---
        if cfg.allowed_sources is not None:
            source_ok = any(
                allowed in doc.source
                for allowed in cfg.allowed_sources
            )
            if not source_ok:
                issues.append(
                    f"Document source '{doc.source}' is not in the allowed sources allowlist. "
                    "This document was not retrieved from an approved knowledge base."
                )

        # --- Size check ---
        if len(doc.content) > cfg.max_single_doc_chars:
            issues.append(
                f"Document content is {len(doc.content)} chars, exceeding the limit of "
                f"{cfg.max_single_doc_chars} chars per chunk. Large chunks may crowd out "
                "other context or facilitate context-stuffing."
            )

        # --- Injection scan ---
        injection_report = detect_injection(doc.content, source_type="indirect")
        if _risk_meets_threshold(injection_report.risk_level, cfg.min_injection_risk):
            signal_summary = "; ".join(
                f"{s.rule_id}({s.confidence.value})" for s in injection_report.signals
            )
            issues.append(
                f"Injection signals detected in document content "
                f"[risk={injection_report.risk_level.value}]: {signal_summary}"
            )

        passed = len(issues) == 0
        doc_results.append(DocumentCheckResult(
            document=doc,
            passed=passed,
            issues=issues,
            injection_report=injection_report,
        ))

        if passed or not cfg.quarantine_on_flag:
            safe.append(doc)
        else:
            flagged.append(doc)

    return RagSecurityResult(
        total_documents=len(documents),
        flagged_documents=flagged,
        safe_documents=safe,
        doc_results=doc_results,
        total_chars=total_chars,
        warnings=warnings,
    )


def sanitize_rag_context(
    documents: list[RagDocument],
    config: Optional[RagSecurityConfig] = None,
) -> list[RagDocument]:
    """
    Validate and return only the safe documents from a RAG context.

    Convenience wrapper around validate_rag_context that returns the
    safe document list directly for inline use in RAG pipelines.

    Args:
        documents: Retrieved document chunks.
        config:    Security configuration.

    Returns:
        List of documents that passed all security checks.
    """
    result = validate_rag_context(documents, config)
    return result.safe_documents
