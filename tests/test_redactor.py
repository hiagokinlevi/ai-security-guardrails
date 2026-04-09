"""
Tests for guardrails/redaction/redactor.py
"""

import pytest

from guardrails.redaction.redactor import (
    RedactionResult,
    redact_sensitive_data,
    redact_with_report,
)


class TestRedactSensitiveData:
    """Tests for the fast-path redact_sensitive_data() function."""

    def test_no_sensitive_data_unchanged(self) -> None:
        text = "The weather in Paris is sunny today."
        assert redact_sensitive_data(text) == text

    def test_email_is_redacted(self) -> None:
        text = "Contact us at support@example.com for help."
        result = redact_sensitive_data(text)
        assert "support@example.com" not in result
        assert "[REDACTED:EMAIL]" in result

    def test_credit_card_is_redacted(self) -> None:
        text = "My Visa card is 4111 1111 1111 1111."
        result = redact_sensitive_data(text)
        assert "4111" not in result
        assert "[REDACTED:CREDIT_CARD]" in result

    def test_ssn_is_redacted(self) -> None:
        text = "Social security number: 123-45-6789"
        result = redact_sensitive_data(text)
        assert "123-45-6789" not in result
        assert "[REDACTED:SSN]" in result

    def test_aws_access_key_is_redacted(self) -> None:
        text = "Key: AKIAIOSFODNN7EXAMPLE"
        result = redact_sensitive_data(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in result
        assert "[REDACTED:AWS_KEY_ID]" in result

    def test_private_key_is_redacted(self) -> None:
        text = (
            "Here is the key:\n"
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA1234567890\n"
            "-----END RSA PRIVATE KEY-----\n"
        )
        result = redact_sensitive_data(text)
        assert "MIIEowIBAAKCAQEA1234567890" not in result
        assert "[REDACTED:PRIVATE_KEY]" in result

    def test_password_assignment_is_redacted(self) -> None:
        text = "password=MyS3cr3tP@ssword"
        result = redact_sensitive_data(text)
        assert "MyS3cr3tP@ssword" not in result

    def test_private_ip_is_redacted(self) -> None:
        text = "Server address: 192.168.1.100"
        result = redact_sensitive_data(text)
        assert "192.168.1.100" not in result
        assert "[REDACTED:PRIVATE_IP]" in result

    def test_public_ip_not_redacted(self) -> None:
        # Public IPs should not be redacted (they are often legitimate content)
        text = "The server is at 8.8.8.8."
        result = redact_sensitive_data(text)
        assert "8.8.8.8" in result

    def test_multiple_patterns_redacted(self) -> None:
        text = "Email: user@example.com, Card: 4111 1111 1111 1111"
        result = redact_sensitive_data(text)
        assert "user@example.com" not in result
        assert "4111" not in result
        assert "[REDACTED:EMAIL]" in result
        assert "[REDACTED:CREDIT_CARD]" in result

    def test_idempotent_on_clean_text(self) -> None:
        text = "Just a normal sentence with no sensitive data."
        assert redact_sensitive_data(text) == text


class TestRedactWithReport:
    """Tests for the redact_with_report() function that returns metadata."""

    def test_returns_redaction_result_type(self) -> None:
        result = redact_with_report("Hello world")
        assert isinstance(result, RedactionResult)

    def test_no_redaction_on_clean_text(self) -> None:
        text = "Clean text with no sensitive data."
        result = redact_with_report(text)
        assert result.redaction_count == 0
        assert result.redacted_types == []
        assert result.redacted_text == text
        assert result.original_text == text

    def test_redaction_count_matches_occurrences(self) -> None:
        # Two email addresses in the text
        text = "From: alice@example.com, To: bob@example.com"
        result = redact_with_report(text)
        assert result.redaction_count >= 2

    def test_redacted_type_is_recorded(self) -> None:
        text = "Send email to admin@example.com"
        result = redact_with_report(text)
        assert "email" in result.redacted_types

    def test_original_text_preserved_in_result(self) -> None:
        text = "api_key=secret123456789abcdefghijklmn"
        result = redact_with_report(text)
        assert result.original_text == text
        assert "secret123456789abcdefghijklmn" not in result.redacted_text

    def test_multiple_types_recorded(self) -> None:
        text = "user@example.com and 4111 1111 1111 1111"
        result = redact_with_report(text)
        assert "email" in result.redacted_types
        assert "credit_card" in result.redacted_types
