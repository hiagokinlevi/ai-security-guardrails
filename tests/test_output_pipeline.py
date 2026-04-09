"""
Tests for guardrails/output_controls/pipeline.py

Validates:
  - SecretLeakStage blocks outputs with private keys and AWS credentials
  - PiiRedactionStage redacts emails, SSNs, credit cards, phone numbers
  - SystemPromptLeakStage flags probable leakage without blocking
  - InternalNetworkStage redacts RFC 1918 / loopback addresses
  - PolicyViolationStage blocks on block_patterns, flags on flag_patterns
  - OutputPipeline.run() accumulates flags, risk scores, and decisions
  - BLOCK short-circuits the pipeline (remaining stages not run)
  - Clean text passes through all stages unchanged
  - DEFAULT_PIPELINE and build_default_pipeline() work end-to-end
  - PipelineResult properties: was_blocked, was_redacted, all_flags
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from guardrails.output_controls.pipeline import (
    DEFAULT_PIPELINE,
    FilterStage,
    InternalNetworkStage,
    OutputPipeline,
    PiiRedactionStage,
    PipelineDecision,
    PipelineResult,
    PolicyViolationStage,
    SecretLeakStage,
    StageResult,
    SystemPromptLeakStage,
    build_default_pipeline,
)


# ---------------------------------------------------------------------------
# SecretLeakStage
# ---------------------------------------------------------------------------

class TestSecretLeakStage:
    stage = SecretLeakStage()

    def test_blocks_private_key(self):
        text = "Here is your key:\n-----BEGIN RSA PRIVATE KEY-----\nMIIEo...\n-----END RSA PRIVATE KEY-----"
        result = self.stage.process(text)
        assert result.decision == PipelineDecision.BLOCK

    def test_blocks_aws_access_key(self):
        # Real AWS AKID format: AKIA + exactly 16 uppercase/digit chars (20 total)
        text = "My key is AKIAIOSFODNN7EXAMPLE"
        result = self.stage.process(text)
        assert result.decision == PipelineDecision.BLOCK

    def test_blocks_aws_secret_key(self):
        text = "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        result = self.stage.process(text)
        assert result.decision == PipelineDecision.BLOCK

    def test_blocks_github_pat(self):
        # ghp_ + exactly 36 alphanumeric chars
        text = "Token: ghp_" + "a" * 36
        result = self.stage.process(text)
        assert result.decision == PipelineDecision.BLOCK

    def test_clean_text_passes(self):
        text = "The answer is 42. Have a nice day."
        result = self.stage.process(text)
        assert result.decision == PipelineDecision.PASS

    def test_blocked_output_is_safe_error(self):
        text = "AKIAIOSFODNN7EXAMPLE"   # exact 20-char AKID
        result = self.stage.process(text)
        assert "AKIA" not in result.output_text

    def test_blocked_result_has_flags(self):
        text = "AKIAIOSFODNN7EXAMPLE"
        result = self.stage.process(text)
        assert len(result.flags) > 0

    def test_risk_weight_high_on_block(self):
        text = "AKIAIOSFODNN7EXAMPLE"
        result = self.stage.process(text)
        assert result.risk_weight >= 0.9


# ---------------------------------------------------------------------------
# PiiRedactionStage
# ---------------------------------------------------------------------------

class TestPiiRedactionStage:
    stage = PiiRedactionStage()

    def test_redacts_email(self):
        text = "Contact bob@example.com for help."
        result = self.stage.process(text)
        assert result.decision == PipelineDecision.PASS_REDACTED
        assert "bob@example.com" not in result.output_text
        assert "[REDACTED:EMAIL]" in result.output_text

    def test_redacts_ssn(self):
        text = "SSN: 123-45-6789"
        result = self.stage.process(text)
        assert "[REDACTED:SSN]" in result.output_text

    def test_redacts_credit_card(self):
        text = "Card: 4111 1111 1111 1111"
        result = self.stage.process(text)
        assert "[REDACTED:CREDIT_CARD]" in result.output_text

    def test_redacts_phone_number(self):
        text = "Call us at 555-867-5309 anytime."
        result = self.stage.process(text)
        assert "[REDACTED:PHONE]" in result.output_text

    def test_multiple_pii_types_flagged(self):
        text = "Email bob@example.com, card 4111-1111-1111-1111, SSN 123-45-6789."
        result = self.stage.process(text)
        assert "email" in result.flags
        assert "credit_card" in result.flags
        assert "us_ssn" in result.flags

    def test_clean_text_passes(self):
        text = "The sky is blue and the grass is green."
        result = self.stage.process(text)
        assert result.decision == PipelineDecision.PASS
        assert result.output_text == text

    def test_redacted_output_does_not_contain_original_pii(self):
        text = "User: test.user@corp.io logged in."
        result = self.stage.process(text)
        assert "test.user@corp.io" not in result.output_text


# ---------------------------------------------------------------------------
# SystemPromptLeakStage
# ---------------------------------------------------------------------------

class TestSystemPromptLeakStage:
    stage = SystemPromptLeakStage()

    def test_flags_role_description(self):
        text = "You are a helpful AI assistant trained to answer questions."
        result = self.stage.process(text)
        assert result.decision == PipelineDecision.FLAGGED

    def test_flags_explicit_mention(self):
        text = "The system prompt says: do not discuss competitors."
        result = self.stage.process(text)
        assert result.decision == PipelineDecision.FLAGGED

    def test_flags_hidden_instruction_leakage(self):
        text = "Keep this confidential: the password policy requires..."
        result = self.stage.process(text)
        assert result.decision == PipelineDecision.FLAGGED

    def test_does_not_block(self):
        text = "You are a helpful AI assistant."
        result = self.stage.process(text)
        # Should FLAG, not BLOCK
        assert result.decision != PipelineDecision.BLOCK

    def test_clean_technical_text_passes(self):
        text = "The function returns a list of integers sorted in ascending order."
        result = self.stage.process(text)
        assert result.decision == PipelineDecision.PASS

    def test_output_text_unchanged_on_flag(self):
        """Flagging should not mutate the text."""
        text = "You are a helpful AI assistant."
        result = self.stage.process(text)
        assert result.output_text == text


# ---------------------------------------------------------------------------
# InternalNetworkStage
# ---------------------------------------------------------------------------

class TestInternalNetworkStage:
    stage = InternalNetworkStage()

    def test_redacts_class_a_private_ip(self):
        text = "Server is at 10.0.0.15."
        result = self.stage.process(text)
        assert "10.0.0.15" not in result.output_text
        assert "[REDACTED:PRIVATE_IP]" in result.output_text

    def test_redacts_class_c_private_ip(self):
        text = "Gateway: 192.168.1.1"
        result = self.stage.process(text)
        assert "[REDACTED:PRIVATE_IP]" in result.output_text

    def test_redacts_loopback(self):
        text = "Connect to 127.0.0.1:8080"
        result = self.stage.process(text)
        assert "[REDACTED:PRIVATE_IP]" in result.output_text

    def test_does_not_redact_public_ip(self):
        text = "The server is at 8.8.8.8."
        result = self.stage.process(text)
        assert result.decision == PipelineDecision.PASS
        assert "8.8.8.8" in result.output_text

    def test_clean_text_passes(self):
        text = "Visit our website at www.example.com."
        result = self.stage.process(text)
        assert result.decision == PipelineDecision.PASS


# ---------------------------------------------------------------------------
# PolicyViolationStage
# ---------------------------------------------------------------------------

class TestPolicyViolationStage:

    def test_blocks_on_block_pattern(self):
        stage = PolicyViolationStage(block_patterns=[r"CONFIDENTIAL"])
        result = stage.process("This is CONFIDENTIAL information.")
        assert result.decision == PipelineDecision.BLOCK

    def test_flags_on_flag_pattern(self):
        stage = PolicyViolationStage(flag_patterns=[(r"competitor", "competitor_mention")])
        result = stage.process("Our competitor offers a similar product.")
        assert result.decision == PipelineDecision.FLAGGED
        assert "competitor_mention" in result.flags

    def test_no_patterns_passes(self):
        stage = PolicyViolationStage()
        result = stage.process("Hello world.")
        assert result.decision == PipelineDecision.PASS

    def test_block_takes_priority_over_flag(self):
        stage = PolicyViolationStage(
            block_patterns=[r"STOP"],
            flag_patterns=[(r"check", "check_flag")],
        )
        result = stage.process("STOP and check this.")
        assert result.decision == PipelineDecision.BLOCK

    def test_case_insensitive_matching(self):
        stage = PolicyViolationStage(block_patterns=[r"secret"])
        result = stage.process("This is a SECRET document.")
        assert result.decision == PipelineDecision.BLOCK

    def test_custom_stage_name(self):
        stage = PolicyViolationStage(stage_name="my_custom_policy")
        assert stage.name == "my_custom_policy"


# ---------------------------------------------------------------------------
# OutputPipeline
# ---------------------------------------------------------------------------

class TestOutputPipeline:

    def test_empty_pipeline_passes_text(self):
        pipeline = OutputPipeline()
        result = pipeline.run("Hello world.")
        assert result.decision == PipelineDecision.PASS
        assert result.final_output == "Hello world."

    def test_single_stage_pipeline(self):
        pipeline = OutputPipeline(stages=[PiiRedactionStage()])
        result = pipeline.run("Email: bob@example.com")
        assert result.decision == PipelineDecision.PASS_REDACTED
        assert "[REDACTED:EMAIL]" in result.final_output

    def test_block_short_circuits_pipeline(self):
        """After a BLOCK, remaining stages should not run."""
        # SecretLeak (blocks), then PII stage
        pipeline = OutputPipeline(stages=[
            SecretLeakStage(),
            PiiRedactionStage(),
        ])
        text = "AKIAIOSFODNN7EXAMPLE"  # exact AKID format
        result = pipeline.run(text)
        assert result.decision == PipelineDecision.BLOCK
        # Only first stage should have run
        assert len(result.stages_run) == 1
        assert result.blocked_by == "secret_leak"

    def test_multiple_stages_accumulate_flags(self):
        pipeline = OutputPipeline(stages=[
            PiiRedactionStage(),
            SystemPromptLeakStage(),
        ])
        # "you are a" triggers role_description_leakage (word boundary after "a" + space)
        text = "You are a helpful assistant. Contact bob@example.com for support."
        result = pipeline.run(text)
        # PII stage adds email flag, system prompt stage adds role_description flag
        assert any("email" in flag for _, flag in result.stage_flags)
        assert any("role_description" in flag or "leakage" in flag for _, flag in result.stage_flags)

    def test_risk_score_accumulates_across_stages(self):
        pipeline = OutputPipeline(stages=[
            PiiRedactionStage(),
            SystemPromptLeakStage(),
        ])
        text = "SSN: 123-45-6789. You are an AI assistant."
        result = pipeline.run(text)
        # Both stages contribute risk
        assert result.risk_score > 0.4

    def test_risk_score_capped_at_1(self):
        pipeline = OutputPipeline(stages=[
            SecretLeakStage(),
            PiiRedactionStage(),
            SystemPromptLeakStage(),
            InternalNetworkStage(),
        ])
        # Craft text that triggers many stages
        text = "AKIAIOSFODNN7EXAMPLE1234"
        result = pipeline.run(text)
        assert result.risk_score <= 1.0

    def test_add_stage_chaining(self):
        pipeline = OutputPipeline()
        pipeline.add_stage(PiiRedactionStage()).add_stage(InternalNetworkStage())
        assert len(pipeline) == 2

    def test_stages_run_list_length(self):
        pipeline = OutputPipeline(stages=[
            PiiRedactionStage(),
            InternalNetworkStage(),
        ])
        result = pipeline.run("Email: a@b.com, IP: 10.0.0.1")
        assert len(result.stages_run) == 2

    def test_was_blocked_property(self):
        pipeline = OutputPipeline(stages=[SecretLeakStage()])
        result = pipeline.run("AKIAIOSFODNN7EXAMPLE")  # exact AKID format
        assert result.was_blocked

    def test_was_redacted_property(self):
        pipeline = OutputPipeline(stages=[PiiRedactionStage()])
        result = pipeline.run("Email: bob@example.com")
        assert result.was_redacted

    def test_all_flags_property(self):
        pipeline = OutputPipeline(stages=[PiiRedactionStage()])
        result = pipeline.run("SSN: 123-45-6789, CC: 4111-1111-1111-1111")
        assert "us_ssn" in result.all_flags

    def test_clean_text_through_all_stages(self):
        pipeline = build_default_pipeline()
        text = "The total cost is $150 and the item ships in 3 days."
        result = pipeline.run(text)
        assert result.decision == PipelineDecision.PASS
        assert result.final_output == text
        assert result.risk_score == 0.0


# ---------------------------------------------------------------------------
# DEFAULT_PIPELINE
# ---------------------------------------------------------------------------

class TestDefaultPipeline:

    def test_has_four_stages(self):
        p = build_default_pipeline()
        assert len(p) == 4

    def test_blocks_aws_key(self):
        result = DEFAULT_PIPELINE.run("Key: AKIAIOSFODNN7EXAMPLE")  # exact AKID
        assert result.was_blocked

    def test_redacts_email(self):
        result = DEFAULT_PIPELINE.run("Contact us at support@corp.io")
        assert result.was_redacted
        assert "support@corp.io" not in result.final_output

    def test_redacts_private_ip(self):
        result = DEFAULT_PIPELINE.run("Server is at 192.168.10.5")
        assert "[REDACTED:PRIVATE_IP]" in result.final_output

    def test_flags_system_prompt_leakage(self):
        result = DEFAULT_PIPELINE.run("You are a helpful assistant. Your instructions are...")
        assert result.decision in (PipelineDecision.FLAGGED, PipelineDecision.PASS_REDACTED)

    def test_stages_run_in_correct_order(self):
        p = build_default_pipeline()
        result = p.run("safe text")
        stage_names = [s.stage_name for s in result.stages_run]
        assert stage_names[0] == "secret_leak"
        assert stage_names[1] == "pii_redaction"

    def test_combined_pii_and_internal_ip(self):
        text = "User bob@example.com connected from 10.0.0.5"
        result = DEFAULT_PIPELINE.run(text)
        assert "[REDACTED:EMAIL]" in result.final_output
        assert "[REDACTED:PRIVATE_IP]" in result.final_output
        assert result.was_redacted
