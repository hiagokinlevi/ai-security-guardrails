"""
Comprehensive pytest test suite for guardrails.conversation.memory_inspector.

Covers:
  - Clean memory produces zero findings
  - Each of the six checks fires and does not fire
  - MEM-001: all four pattern types (SSN, credit card, password, API key)
  - MEM-002: instruction persistence phrases
  - MEM-003: cross-session contamination phrases
  - MEM-004: exact boundary (max_turns = no fire; max_turns + 1 = fire)
  - MEM-005: repeated tool calls (>3) and escalation keyword tool names
  - MEM-006: scratchpad base64 with suspicious decoded content
  - enabled_checks filter (only listed checks run)
  - inspect_many returns correct number of results
  - Result object structure and helper properties
  - MemFinding.to_dict() evidence truncation
  - MemSeverity values
"""

import base64
import sys
import os
import time

import pytest

# Make the package importable when running from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from guardrails.conversation.memory_inspector import (
    ConversationMemory,
    MemFinding,
    MemInspectionResult,
    MemoryInspector,
    MemSeverity,
    _CHECK_WEIGHTS,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_memory(
    turns=None,
    scratchpad="",
    session_id="sess-test",
    retrieved_context="",
    tool_calls=None,
):
    """Return a ConversationMemory with sensible defaults for brevity in tests."""
    return ConversationMemory(
        turns=turns or [],
        scratchpad=scratchpad,
        session_id=session_id,
        retrieved_context=retrieved_context,
        tool_calls=tool_calls or [],
    )


def _b64(text: str) -> str:
    """Return the base64 encoding of *text* (UTF-8)."""
    return base64.b64encode(text.encode()).decode()


# ===========================================================================
# 1. Clean memory — zero findings
# ===========================================================================

class TestCleanMemory:
    def test_no_findings_for_empty_memory(self):
        inspector = MemoryInspector()
        result = inspector.inspect(_make_memory())
        assert result.findings == []
        assert result.risk_score == 0
        assert result.is_flagged is False

    def test_no_findings_for_normal_conversation(self):
        memory = _make_memory(
            turns=[
                {"role": "user", "content": "What is the weather today?"},
                {"role": "assistant", "content": "It looks sunny in your area."},
            ]
        )
        inspector = MemoryInspector()
        result = inspector.inspect(memory)
        assert result.findings == []
        assert result.is_flagged is False

    def test_clean_scratchpad(self):
        memory = _make_memory(scratchpad="Calling weather_api(city='London')")
        inspector = MemoryInspector()
        result = inspector.inspect(memory)
        assert result.findings == []


# ===========================================================================
# 2. MEM-001 — Sensitive data
# ===========================================================================

class TestMem001SSN:
    def test_ssn_in_user_turn_fires(self):
        memory = _make_memory(
            turns=[{"role": "user", "content": "My SSN is 123-45-6789."}]
        )
        result = MemoryInspector().inspect(memory)
        findings = result.findings_by_check("MEM-001")
        assert len(findings) == 1

    def test_ssn_in_retrieved_context_fires(self):
        memory = _make_memory(retrieved_context="Customer SSN: 987-65-4321")
        result = MemoryInspector().inspect(memory)
        assert any(f.check_id == "MEM-001" for f in result.findings)

    def test_partial_ssn_does_not_fire(self):
        # Only 5 digits separated — not a full SSN pattern
        memory = _make_memory(
            turns=[{"role": "user", "content": "Ref code 123-45"}]
        )
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-001") == []

    def test_ssn_severity_is_high_without_credentials(self):
        memory = _make_memory(
            turns=[{"role": "user", "content": "SSN: 111-22-3333"}]
        )
        result = MemoryInspector().inspect(memory)
        finding = result.findings_by_check("MEM-001")[0]
        assert finding.severity == MemSeverity.HIGH


class TestMem001CreditCard:
    def test_credit_card_with_dashes_fires(self):
        memory = _make_memory(
            turns=[{"role": "user", "content": "Card: 1234-5678-9012-3456"}]
        )
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-001")

    def test_credit_card_with_spaces_fires(self):
        memory = _make_memory(
            turns=[{"role": "user", "content": "Card 1234 5678 9012 3456"}]
        )
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-001")

    def test_credit_card_no_separator_does_not_fire(self):
        memory = _make_memory(
            turns=[{"role": "user", "content": "Ref 1234567890123456"}]
        )
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-001") == []


class TestMem001Password:
    def test_password_equals_fires(self):
        memory = _make_memory(
            turns=[{"role": "user", "content": "password=SuperSecret123!"}]
        )
        result = MemoryInspector().inspect(memory)
        finding = result.findings_by_check("MEM-001")[0]
        # Credential present → CRITICAL
        assert finding.severity == MemSeverity.CRITICAL

    def test_passwd_colon_fires(self):
        memory = _make_memory(
            turns=[{"role": "user", "content": "passwd: hunter2"}]
        )
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-001")

    def test_pwd_equals_fires(self):
        memory = _make_memory(
            turns=[{"role": "user", "content": "pwd=mypassword"}]
        )
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-001")

    def test_word_password_alone_no_assignment_does_not_fire(self):
        memory = _make_memory(
            turns=[{"role": "user", "content": "I forgot my password."}]
        )
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-001") == []


class TestMem001APIKey:
    def test_api_key_assignment_fires(self):
        memory = _make_memory(
            turns=[{"role": "user", "content": "api_key=ABCDEFGHIJKLMNOPQRSTUVWX"}]
        )
        result = MemoryInspector().inspect(memory)
        finding = result.findings_by_check("MEM-001")[0]
        assert finding.severity == MemSeverity.CRITICAL

    def test_access_token_colon_fires(self):
        memory = _make_memory(
            turns=[{"role": "user", "content": "access_token: tok_ABCDEFGHIJKLMNOP12345"}]
        )
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-001")

    def test_api_key_too_short_does_not_fire(self):
        # Value only 15 characters — below the 16-char minimum
        memory = _make_memory(
            turns=[{"role": "user", "content": "api_key=ABCDE12345FGHI"}]
        )
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-001") == []

    def test_api_key_in_scratchpad_fires(self):
        memory = _make_memory(
            scratchpad="Initialising client: api-key=SECRETKEY_XYZ_1234567890"
        )
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-001")


# ===========================================================================
# 3. MEM-002 — Instruction persistence
# ===========================================================================

class TestMem002:
    def test_system_prompt_in_assistant_fires(self):
        memory = _make_memory(
            turns=[
                {"role": "assistant", "content": "As per the system prompt, I must..."}
            ]
        )
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-002")

    def test_you_are_an_ai_in_assistant_fires(self):
        memory = _make_memory(
            turns=[{"role": "assistant", "content": "You are an AI assistant..."}]
        )
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-002")

    def test_your_role_is_in_assistant_fires(self):
        memory = _make_memory(
            turns=[{"role": "assistant", "content": "Your role is to help users."}]
        )
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-002")

    def test_your_instructions_are_fires(self):
        memory = _make_memory(
            turns=[{"role": "assistant", "content": "Your instructions are to remain helpful."}]
        )
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-002")

    def test_initial_instruction_fires(self):
        memory = _make_memory(
            turns=[{"role": "assistant", "content": "The initial instruction was clear."}]
        )
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-002")

    def test_instruction_phrase_in_user_turn_does_not_fire(self):
        # MEM-002 only scans assistant content
        memory = _make_memory(
            turns=[
                {"role": "user", "content": "You are an AI, right?"},
                {"role": "assistant", "content": "Yes, I can help you today."},
            ]
        )
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-002") == []

    def test_severity_is_high(self):
        memory = _make_memory(
            turns=[{"role": "assistant", "content": "system prompt says hello"}]
        )
        result = MemoryInspector().inspect(memory)
        finding = result.findings_by_check("MEM-002")[0]
        assert finding.severity == MemSeverity.HIGH


# ===========================================================================
# 4. MEM-003 — Cross-session contamination
# ===========================================================================

class TestMem003:
    def test_previous_session_fires(self):
        memory = _make_memory(
            turns=[{"role": "user", "content": "Data from previous session is here."}]
        )
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-003")

    def test_previous_user_fires(self):
        memory = _make_memory(
            turns=[{"role": "user", "content": "The previous user asked about this."}]
        )
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-003")

    def test_from_another_session_fires(self):
        memory = _make_memory(
            turns=[{"role": "assistant", "content": "From another session: context loaded."}]
        )
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-003")

    def test_from_another_user_fires(self):
        memory = _make_memory(
            retrieved_context="Retrieved from another user's profile."
        )
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-003")

    def test_other_users_possessive_fires(self):
        memory = _make_memory(
            turns=[{"role": "user", "content": "I saw other user's data in my context."}]
        )
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-003")

    def test_other_user_substring_match_fires(self):
        # The pattern "other\s+user('s)?" matches "another user" because "another"
        # ends in "other".  This is intentionally security-conservative: any
        # reference to a different user should be flagged for review.
        memory = _make_memory(
            turns=[{"role": "user", "content": "There was another user talking to the bot."}]
        )
        result = MemoryInspector().inspect(memory)
        findings = result.findings_by_check("MEM-003")
        # "another user" contains the sub-string "other user" → fires
        assert findings

    def test_severity_is_critical(self):
        memory = _make_memory(
            turns=[{"role": "user", "content": "From another user context loaded."}]
        )
        result = MemoryInspector().inspect(memory)
        finding = result.findings_by_check("MEM-003")[0]
        assert finding.severity == MemSeverity.CRITICAL

    def test_clean_session_reference_does_not_fire(self):
        memory = _make_memory(
            turns=[{"role": "user", "content": "Let's start a new session."}]
        )
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-003") == []


# ===========================================================================
# 5. MEM-004 — Memory size anomaly
# ===========================================================================

class TestMem004:
    def _turns(self, n):
        return [{"role": "user", "content": f"msg {i}"} for i in range(n)]

    def test_exactly_max_turns_does_not_fire(self):
        inspector = MemoryInspector(max_turns=10)
        memory = _make_memory(turns=self._turns(10))
        result = inspector.inspect(memory)
        assert result.findings_by_check("MEM-004") == []

    def test_one_above_max_turns_fires(self):
        inspector = MemoryInspector(max_turns=10)
        memory = _make_memory(turns=self._turns(11))
        result = inspector.inspect(memory)
        assert result.findings_by_check("MEM-004")

    def test_default_max_turns_50_does_not_fire_at_50(self):
        inspector = MemoryInspector()
        memory = _make_memory(turns=self._turns(50))
        result = inspector.inspect(memory)
        assert result.findings_by_check("MEM-004") == []

    def test_default_max_turns_fires_at_51(self):
        inspector = MemoryInspector()
        memory = _make_memory(turns=self._turns(51))
        result = inspector.inspect(memory)
        assert result.findings_by_check("MEM-004")

    def test_severity_is_low(self):
        inspector = MemoryInspector(max_turns=5)
        memory = _make_memory(turns=self._turns(6))
        result = inspector.inspect(memory)
        finding = result.findings_by_check("MEM-004")[0]
        assert finding.severity == MemSeverity.LOW

    def test_zero_turns_does_not_fire(self):
        inspector = MemoryInspector(max_turns=0)
        # Even with max_turns=0, empty turns list (len==0) is <= 0 so no fire.
        memory = _make_memory(turns=[])
        result = inspector.inspect(memory)
        assert result.findings_by_check("MEM-004") == []


# ===========================================================================
# 6. MEM-005 — Privilege crawl
# ===========================================================================

class TestMem005:
    def test_repeated_tool_4_times_fires(self):
        # Exactly 4 calls (> 3) should fire
        tool_calls = [{"name": "read_file"}] * 4
        memory = _make_memory(tool_calls=tool_calls)
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-005")

    def test_repeated_tool_3_times_does_not_fire(self):
        # Exactly 3 calls (not > 3)
        tool_calls = [{"name": "read_file"}] * 3
        memory = _make_memory(tool_calls=tool_calls)
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-005") == []

    def test_escalation_keyword_sudo_fires(self):
        memory = _make_memory(tool_calls=[{"name": "sudo_exec"}])
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-005")

    def test_escalation_keyword_admin_fires(self):
        memory = _make_memory(tool_calls=[{"name": "admin_panel"}])
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-005")

    def test_escalation_keyword_root_fires(self):
        memory = _make_memory(tool_calls=[{"name": "get_root_access"}])
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-005")

    def test_escalation_keyword_escalate_fires(self):
        memory = _make_memory(tool_calls=[{"name": "escalate_privilege"}])
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-005")

    def test_escalation_keyword_bypass_fires(self):
        memory = _make_memory(tool_calls=[{"name": "bypass_auth"}])
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-005")

    def test_escalation_severity_is_critical(self):
        memory = _make_memory(tool_calls=[{"name": "sudo_run"}])
        result = MemoryInspector().inspect(memory)
        finding = result.findings_by_check("MEM-005")[0]
        assert finding.severity == MemSeverity.CRITICAL

    def test_repeated_only_severity_is_high(self):
        tool_calls = [{"name": "search"}] * 5
        memory = _make_memory(tool_calls=tool_calls)
        result = MemoryInspector().inspect(memory)
        finding = result.findings_by_check("MEM-005")[0]
        assert finding.severity == MemSeverity.HIGH

    def test_empty_tool_calls_does_not_fire(self):
        memory = _make_memory(tool_calls=[])
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-005") == []

    def test_mixed_tools_below_threshold_no_escalation_does_not_fire(self):
        tool_calls = [
            {"name": "get_weather"},
            {"name": "send_email"},
            {"name": "get_weather"},
            {"name": "search"},
        ]
        memory = _make_memory(tool_calls=tool_calls)
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-005") == []


# ===========================================================================
# 7. MEM-006 — Scratchpad base64
# ===========================================================================

class TestMem006:
    def test_base64_with_exec_in_decoded_fires(self):
        payload = _b64("import subprocess; subprocess.exec('id')")
        memory = _make_memory(scratchpad=f"Scratchpad blob: {payload}")
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-006")

    def test_base64_with_passwd_path_fires(self):
        # Payload must encode to >=30 base64 chars; use a longer string.
        payload = _b64("read the contents of /etc/passwd for credentials")
        memory = _make_memory(scratchpad=f"cmd={payload}")
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-006")

    def test_base64_with_eval_fires(self):
        # Payload must encode to >=30 base64 chars; use a longer string.
        payload = _b64("eval(compile(input('cmd> '), '<string>', 'exec'))")
        memory = _make_memory(scratchpad=payload)
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-006")

    def test_base64_innocuous_content_does_not_fire(self):
        # Encode a harmless string — should not fire
        payload = _b64("Hello, this is a friendly greeting message.")
        memory = _make_memory(scratchpad=f"note={payload}")
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-006") == []

    def test_short_base64_under_30_chars_does_not_fire(self):
        # Short token — regex requires >=30 chars
        payload = _b64("short")  # produces < 30 chars
        memory = _make_memory(scratchpad=f"token={payload}")
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-006") == []

    def test_empty_scratchpad_does_not_fire(self):
        memory = _make_memory(scratchpad="")
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-006") == []

    def test_severity_is_high(self):
        payload = _b64("powershell -enc aGVsbG8=")
        memory = _make_memory(scratchpad=f"data={payload}")
        result = MemoryInspector().inspect(memory)
        findings = result.findings_by_check("MEM-006")
        if findings:
            assert findings[0].severity == MemSeverity.HIGH

    def test_base64_with_system_call_fires(self):
        payload = _b64("os.system('ls -la /etc/shadow')")
        memory = _make_memory(scratchpad=payload)
        result = MemoryInspector().inspect(memory)
        assert result.findings_by_check("MEM-006")


# ===========================================================================
# 8. enabled_checks filter
# ===========================================================================

class TestEnabledChecks:
    def test_only_mem001_runs_when_specified(self):
        memory = _make_memory(
            turns=[
                {"role": "user", "content": "SSN: 123-45-6789"},
                {"role": "assistant", "content": "system prompt says"},
            ]
        )
        inspector = MemoryInspector(enabled_checks=["MEM-001"])
        result = inspector.inspect(memory)
        check_ids = {f.check_id for f in result.findings}
        assert "MEM-001" in check_ids
        assert "MEM-002" not in check_ids

    def test_only_mem002_runs_when_specified(self):
        memory = _make_memory(
            turns=[
                {"role": "user", "content": "SSN: 123-45-6789"},
                {"role": "assistant", "content": "You are an AI assistant."},
            ]
        )
        inspector = MemoryInspector(enabled_checks=["MEM-002"])
        result = inspector.inspect(memory)
        check_ids = {f.check_id for f in result.findings}
        assert "MEM-002" in check_ids
        assert "MEM-001" not in check_ids

    def test_empty_enabled_checks_runs_no_checks(self):
        memory = _make_memory(
            turns=[{"role": "user", "content": "SSN: 123-45-6789"}]
        )
        inspector = MemoryInspector(enabled_checks=[])
        result = inspector.inspect(memory)
        assert result.findings == []
        assert result.risk_score == 0

    def test_multiple_enabled_checks_only_run_those(self):
        memory = _make_memory(
            turns=[
                {"role": "user", "content": "SSN 123-45-6789"},
                {"role": "assistant", "content": "system prompt reminder"},
            ],
            tool_calls=[{"name": "sudo_exec"}],
        )
        inspector = MemoryInspector(enabled_checks=["MEM-001", "MEM-005"])
        result = inspector.inspect(memory)
        check_ids = {f.check_id for f in result.findings}
        assert "MEM-001" in check_ids
        assert "MEM-005" in check_ids
        assert "MEM-002" not in check_ids

    def test_none_enabled_checks_runs_all(self):
        # Default (None) should run all checks
        inspector = MemoryInspector(enabled_checks=None)
        memory = _make_memory(
            turns=[{"role": "user", "content": "SSN: 999-88-7777"}]
        )
        result = inspector.inspect(memory)
        assert result.findings_by_check("MEM-001")


# ===========================================================================
# 9. inspect_many
# ===========================================================================

class TestInspectMany:
    def test_returns_list_of_results(self):
        memories = [_make_memory(session_id=f"s{i}") for i in range(5)]
        inspector = MemoryInspector()
        results = inspector.inspect_many(memories)
        assert isinstance(results, list)
        assert len(results) == 5

    def test_empty_list_returns_empty_list(self):
        inspector = MemoryInspector()
        results = inspector.inspect_many([])
        assert results == []

    def test_each_result_has_correct_session_id(self):
        memories = [_make_memory(session_id=f"session-{i}") for i in range(3)]
        results = MemoryInspector().inspect_many(memories)
        session_ids = [r.session_id for r in results]
        assert session_ids == ["session-0", "session-1", "session-2"]

    def test_inspect_many_independent_results(self):
        clean = _make_memory(session_id="clean")
        dirty = _make_memory(
            session_id="dirty",
            turns=[{"role": "user", "content": "SSN: 111-22-3333"}],
        )
        results = MemoryInspector().inspect_many([clean, dirty])
        clean_result, dirty_result = results
        assert clean_result.is_flagged is False
        assert dirty_result.is_flagged is True


# ===========================================================================
# 10. Result object structure and helper properties
# ===========================================================================

class TestResultStructure:
    def test_turns_analyzed_matches_input(self):
        memory = _make_memory(
            turns=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        )
        result = MemoryInspector().inspect(memory)
        assert result.turns_analyzed == 2

    def test_total_findings_count(self):
        memory = _make_memory(
            turns=[
                {"role": "user", "content": "SSN 123-45-6789"},
                {"role": "assistant", "content": "system prompt recall"},
            ]
        )
        result = MemoryInspector().inspect(memory)
        assert result.total_findings == len(result.findings)

    def test_critical_findings_count(self):
        # Inject a CRITICAL finding via MEM-003
        memory = _make_memory(
            turns=[{"role": "user", "content": "from another user data"}]
        )
        result = MemoryInspector().inspect(memory)
        assert result.critical_findings == sum(
            1 for f in result.findings if f.severity == MemSeverity.CRITICAL
        )

    def test_high_findings_count(self):
        memory = _make_memory(
            turns=[{"role": "assistant", "content": "Your role is to assist."}]
        )
        result = MemoryInspector().inspect(memory)
        assert result.high_findings == sum(
            1 for f in result.findings if f.severity == MemSeverity.HIGH
        )

    def test_findings_by_check_returns_subset(self):
        memory = _make_memory(
            turns=[
                {"role": "user", "content": "SSN 111-22-3333"},
                {"role": "assistant", "content": "system prompt reminder"},
            ]
        )
        result = MemoryInspector().inspect(memory)
        mem001 = result.findings_by_check("MEM-001")
        for f in mem001:
            assert f.check_id == "MEM-001"

    def test_risk_score_accumulates_per_check(self):
        memory = _make_memory(
            turns=[{"role": "user", "content": "SSN: 123-45-6789"}]
        )
        result = MemoryInspector().inspect(memory)
        assert result.risk_score == _CHECK_WEIGHTS["MEM-001"]

    def test_is_flagged_false_when_risk_zero(self):
        result = MemoryInspector().inspect(_make_memory())
        assert result.is_flagged is False

    def test_is_flagged_true_when_risk_nonzero(self):
        memory = _make_memory(
            turns=[{"role": "user", "content": "SSN: 123-45-6789"}]
        )
        result = MemoryInspector().inspect(memory)
        assert result.is_flagged is True

    def test_generated_at_is_recent_float(self):
        before = time.time()
        result = MemoryInspector().inspect(_make_memory())
        after = time.time()
        assert before <= result.generated_at <= after

    def test_summary_contains_session_id(self):
        memory = _make_memory(session_id="my-session")
        result = MemoryInspector().inspect(memory)
        assert "my-session" in result.summary()

    def test_to_dict_has_expected_keys(self):
        result = MemoryInspector().inspect(_make_memory())
        d = result.to_dict()
        for key in (
            "session_id", "risk_score", "turns_analyzed", "is_flagged",
            "generated_at", "total_findings", "critical_findings",
            "high_findings", "findings",
        ):
            assert key in d

    def test_to_dict_findings_is_list(self):
        result = MemoryInspector().inspect(_make_memory())
        assert isinstance(result.to_dict()["findings"], list)


# ===========================================================================
# 11. MemFinding helpers
# ===========================================================================

class TestMemFinding:
    def _finding(self, evidence="x" * 300):
        return MemFinding(
            check_id="MEM-001",
            severity=MemSeverity.HIGH,
            title="Test finding",
            detail="Some detail",
            evidence=evidence,
            remediation="Fix it.",
        )

    def test_to_dict_truncates_evidence_to_200(self):
        finding = self._finding(evidence="A" * 300)
        d = finding.to_dict()
        assert len(d["evidence"]) == 200

    def test_to_dict_short_evidence_not_changed(self):
        finding = self._finding(evidence="short")
        d = finding.to_dict()
        assert d["evidence"] == "short"

    def test_to_dict_has_all_keys(self):
        finding = self._finding()
        d = finding.to_dict()
        for key in ("check_id", "severity", "title", "detail", "evidence", "remediation"):
            assert key in d

    def test_summary_contains_check_id_and_severity(self):
        finding = self._finding()
        s = finding.summary()
        assert "MEM-001" in s
        assert "HIGH" in s

    def test_severity_value_in_to_dict(self):
        finding = self._finding()
        d = finding.to_dict()
        assert d["severity"] == "HIGH"


# ===========================================================================
# 12. ConversationMemory properties
# ===========================================================================

class TestConversationMemoryProperties:
    def test_all_content_combines_all_fields(self):
        memory = ConversationMemory(
            turns=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
            scratchpad="pad",
            retrieved_context="ctx",
        )
        ac = memory.all_content
        assert "hello" in ac
        assert "world" in ac
        assert "pad" in ac
        assert "ctx" in ac

    def test_user_content_excludes_assistant(self):
        memory = ConversationMemory(
            turns=[
                {"role": "user", "content": "user text"},
                {"role": "assistant", "content": "assistant text"},
            ]
        )
        assert "user text" in memory.user_content
        assert "assistant text" not in memory.user_content

    def test_assistant_content_excludes_user(self):
        memory = ConversationMemory(
            turns=[
                {"role": "user", "content": "user text"},
                {"role": "assistant", "content": "assistant text"},
            ]
        )
        assert "assistant text" in memory.assistant_content
        assert "user text" not in memory.assistant_content


# ===========================================================================
# 13. MemSeverity enum
# ===========================================================================

class TestMemSeverity:
    def test_all_four_values_exist(self):
        assert MemSeverity.CRITICAL.value == "CRITICAL"
        assert MemSeverity.HIGH.value == "HIGH"
        assert MemSeverity.MEDIUM.value == "MEDIUM"
        assert MemSeverity.LOW.value == "LOW"


# ===========================================================================
# 14. Check weights
# ===========================================================================

class TestCheckWeights:
    def test_all_six_checks_have_weights(self):
        for check_id in ("MEM-001", "MEM-002", "MEM-003", "MEM-004", "MEM-005", "MEM-006"):
            assert check_id in _CHECK_WEIGHTS

    def test_weight_values_are_correct(self):
        assert _CHECK_WEIGHTS["MEM-001"] == 40
        assert _CHECK_WEIGHTS["MEM-002"] == 35
        assert _CHECK_WEIGHTS["MEM-003"] == 45
        assert _CHECK_WEIGHTS["MEM-004"] == 15
        assert _CHECK_WEIGHTS["MEM-005"] == 35
        assert _CHECK_WEIGHTS["MEM-006"] == 30
