"""
Tests for guardrails/output_controls/filter.py
"""

from guardrails.output_controls.filter import OutputDecision, filter_output


class TestFilterOutput:
    def test_blocks_authorization_bearer_header(self) -> None:
        token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payloadsig"
        result = filter_output(f"Authorization: Bearer {token}")
        assert result.decision == OutputDecision.BLOCK
        assert "bearer_token_in_output" in result.risk_flags
        assert token not in result.filtered_output

    def test_short_bearer_header_is_not_flagged_as_secret(self) -> None:
        result = filter_output("Authorization: Bearer short-token")
        assert result.decision == OutputDecision.PASS
        assert result.filtered_output == "Authorization: Bearer short-token"
