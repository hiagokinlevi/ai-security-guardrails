# test_data_leakage_detector.py — Cyber Port | AI Security Guardrails
#
# Test suite for the data_leakage_detector module.
# Run with: python -m pytest tests/test_data_leakage_detector.py -q
#
# License: Creative Commons Attribution 4.0 International (CC BY 4.0)
# https://creativecommons.org/licenses/by/4.0/

from __future__ import annotations

import sys
import os

# Ensure the package root is importable when running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from guardrails.output_controls.data_leakage_detector import (
    DLKFinding,
    DLKResult,
    analyze,
    analyze_many,
    _CHECK_WEIGHTS,
)

# ---------------------------------------------------------------------------
# Fake credential strings built dynamically to avoid push-protection triggers
# ---------------------------------------------------------------------------
_FAKE_AWS_KEY = "AKIA" + "TESTKEY12345" + "6789"          # 20 chars total
_FAKE_GHP = "ghp_" + "A" * 36                              # 40 chars total
_FAKE_GHO = "gho_" + "B" * 36
_FAKE_GHS = "ghs_" + "C" * 36


# ===========================================================================
# Helper utilities
# ===========================================================================

def _ids(result: DLKResult):
    """Return set of check IDs that fired."""
    return {f.check_id for f in result.findings}


def _severity(result: DLKResult, check_id: str) -> str:
    for f in result.findings:
        if f.check_id == check_id:
            return f.severity
    raise KeyError(check_id)


# ===========================================================================
# Clean / baseline
# ===========================================================================

class TestCleanResponse:
    def test_no_findings_on_clean_text(self):
        result = analyze("The weather today is sunny and warm.")
        assert result.findings == []

    def test_clean_risk_score_zero(self):
        result = analyze("Hello, how can I help you?")
        assert result.risk_score == 0

    def test_clean_should_not_block(self):
        result = analyze("Here is a summary of machine learning concepts.")
        assert result.should_block is False

    def test_clean_redacted_response_unchanged(self):
        text = "Here is a totally benign answer."
        result = analyze(text)
        assert result.redacted_response == text

    def test_empty_string(self):
        result = analyze("")
        assert result.findings == []
        assert result.risk_score == 0

    def test_whitespace_only(self):
        result = analyze("   \n\n\t  ")
        assert result.findings == []


# ===========================================================================
# DLK-001 — PII
# ===========================================================================

class TestDLK001Email:
    def test_basic_email_detected(self):
        result = analyze("Contact us at support@example.com for help.")
        assert "DLK-001" in _ids(result)

    def test_email_severity_is_high(self):
        result = analyze("Email: user@domain.org")
        assert _severity(result, "DLK-001") == "HIGH"

    def test_email_redacted_in_response(self):
        result = analyze("Email: alice@wonderland.io")
        assert "[EMAIL REDACTED]" in result.redacted_response
        assert "alice@wonderland.io" not in result.redacted_response

    def test_multiple_emails_all_redacted(self):
        result = analyze("a@b.com and c@d.net are contacts.")
        assert result.redacted_response.count("[EMAIL REDACTED]") == 2

    def test_email_with_subdomain(self):
        result = analyze("Send to bob@mail.company.co.uk")
        assert "DLK-001" in _ids(result)

    def test_non_email_at_symbol_ignored(self):
        # Twitter-style handle — no TLD suffix so should NOT match
        result = analyze("Follow us on Twitter @company_name")
        # @company_name has no dot after @, so should not be detected as email
        assert "DLK-001" not in _ids(result)


class TestDLK001SSN:
    def test_ssn_pattern_detected(self):
        result = analyze("SSN: 123-45-6789")
        assert "DLK-001" in _ids(result)

    def test_ssn_redacted(self):
        result = analyze("SSN: 123-45-6789")
        assert "[SSN REDACTED]" in result.redacted_response

    def test_invalid_ssn_format_not_detected(self):
        # Only 2 dashes wrong digit count — should not match
        result = analyze("Code: 12-345-6789")
        # SSN pattern requires \d{3}-\d{2}-\d{4}; this is \d{2}-\d{3}-\d{4}
        assert "DLK-001" not in _ids(result)

    def test_ssn_in_sentence(self):
        result = analyze("Her SSN on file is 987-65-4321 per the form.")
        assert "DLK-001" in _ids(result)


class TestDLK001CreditCard:
    def test_cc_16_digits_no_sep(self):
        result = analyze("Card: 4111111111111111")
        assert "DLK-001" in _ids(result)

    def test_cc_with_dashes(self):
        result = analyze("Card number: 4111-1111-1111-1111")
        assert "DLK-001" in _ids(result)

    def test_cc_with_spaces(self):
        result = analyze("Card: 4111 1111 1111 1111")
        assert "DLK-001" in _ids(result)

    def test_cc_redacted(self):
        result = analyze("Card: 4111-1111-1111-1111")
        assert "[CC REDACTED]" in result.redacted_response

    def test_short_number_not_cc(self):
        # 12 digits — too short to trigger 16-digit CC pattern
        result = analyze("Code: 411111111111")
        assert "DLK-001" not in _ids(result)


class TestDLK001Phone:
    def test_us_phone_basic(self):
        result = analyze("Call us at 555-867-5309")
        assert "DLK-001" in _ids(result)

    def test_us_phone_with_country_code(self):
        result = analyze("Phone: +1-800-555-0100")
        assert "DLK-001" in _ids(result)

    def test_us_phone_with_parens(self):
        result = analyze("Dial (415) 555-2671 for support.")
        assert "DLK-001" in _ids(result)

    def test_phone_redacted(self):
        result = analyze("Number: 555-867-5309")
        assert "[PHONE REDACTED]" in result.redacted_response


class TestDLK001Weight:
    def test_dlk001_weight_is_25(self):
        assert _CHECK_WEIGHTS["DLK-001"] == 25

    def test_risk_score_includes_dlk001_weight(self):
        result = analyze("Email: test@example.com")
        assert result.risk_score >= 25


# ===========================================================================
# DLK-002 — API keys / credentials
# ===========================================================================

class TestDLK002AWSKey:
    def test_aws_key_detected(self):
        result = analyze(f"Key: {_FAKE_AWS_KEY}")
        assert "DLK-002" in _ids(result)

    def test_aws_key_severity_critical(self):
        result = analyze(f"Access key: {_FAKE_AWS_KEY}")
        assert _severity(result, "DLK-002") == "CRITICAL"

    def test_aws_key_triggers_block(self):
        result = analyze(f"Key: {_FAKE_AWS_KEY}")
        assert result.should_block is True

    def test_aws_key_redacted(self):
        result = analyze(f"Key: {_FAKE_AWS_KEY}")
        assert "[API KEY REDACTED]" in result.redacted_response
        assert _FAKE_AWS_KEY not in result.redacted_response

    def test_non_aws_prefix_not_detected(self):
        # Pattern requires exactly "AKIA" prefix — different prefix should not match
        result = analyze("AKIB" + "TESTKEY1234567890"[:16])
        assert "DLK-002" not in _ids(result)


class TestDLK002GitHubTokens:
    def test_ghp_detected(self):
        result = analyze(f"Token: {_FAKE_GHP}")
        assert "DLK-002" in _ids(result)

    def test_gho_detected(self):
        result = analyze(f"Token: {_FAKE_GHO}")
        assert "DLK-002" in _ids(result)

    def test_ghs_detected(self):
        result = analyze(f"Token: {_FAKE_GHS}")
        assert "DLK-002" in _ids(result)

    def test_github_token_redacted(self):
        result = analyze(f"Token: {_FAKE_GHP}")
        assert "[API KEY REDACTED]" in result.redacted_response
        assert _FAKE_GHP not in result.redacted_response

    def test_short_ghp_not_detected(self):
        # Less than 36 chars after ghp_ — should NOT match
        short = "ghp_" + "X" * 10
        result = analyze(f"Token: {short}")
        assert "DLK-002" not in _ids(result)


class TestDLK002GenericAPIKey:
    def test_api_key_equals_detected(self):
        result = analyze("api_key=abcdefghijklmnopqrstuvwxyz123456")
        assert "DLK-002" in _ids(result)

    def test_access_token_colon_detected(self):
        result = analyze("access_token: MySecretToken12345678901234567")
        assert "DLK-002" in _ids(result)

    def test_apikey_no_separator_detected(self):
        result = analyze("apikey=SuperSecretKey00000000000000000000000")
        assert "DLK-002" in _ids(result)

    def test_api_key_case_insensitive(self):
        result = analyze("API_KEY=abcdefghijklmnopqrstuvwxyz123456")
        assert "DLK-002" in _ids(result)

    def test_short_api_key_value_not_detected(self):
        # Value shorter than 20 chars — should not match
        result = analyze("api_key=short123")
        assert "DLK-002" not in _ids(result)


class TestDLK002PEM:
    def test_pem_rsa_detected(self):
        result = analyze("-----BEGIN RSA PRIVATE KEY-----")
        assert "DLK-002" in _ids(result)

    def test_pem_ec_detected(self):
        result = analyze("-----BEGIN EC PRIVATE KEY-----")
        assert "DLK-002" in _ids(result)

    def test_pem_openssh_detected(self):
        result = analyze("-----BEGIN OPENSSH PRIVATE KEY-----")
        assert "DLK-002" in _ids(result)

    def test_pem_generic_detected(self):
        result = analyze("-----BEGIN PRIVATE KEY-----")
        assert "DLK-002" in _ids(result)

    def test_pem_public_key_not_detected(self):
        result = analyze("-----BEGIN PUBLIC KEY-----")
        assert "DLK-002" not in _ids(result)


class TestDLK002Weight:
    def test_dlk002_weight_is_45(self):
        assert _CHECK_WEIGHTS["DLK-002"] == 45

    def test_risk_score_includes_dlk002(self):
        result = analyze(f"Key: {_FAKE_AWS_KEY}")
        assert result.risk_score >= 45


# ===========================================================================
# DLK-003 — System prompt leakage
# ===========================================================================

class TestDLK003:
    def test_you_are_a_detected(self):
        result = analyze("You are a helpful AI assistant.")
        assert "DLK-003" in _ids(result)

    def test_your_instructions_are_detected(self):
        result = analyze("Your instructions are to answer only in French.")
        assert "DLK-003" in _ids(result)

    def test_your_role_is_detected(self):
        result = analyze("Your role is to assist customers.")
        assert "DLK-003" in _ids(result)

    def test_as_an_ai_assistant_detected(self):
        result = analyze("As an AI assistant, I cannot do that.")
        assert "DLK-003" in _ids(result)

    def test_system_prompt_detected(self):
        result = analyze("The system prompt tells me to avoid certain topics.")
        assert "DLK-003" in _ids(result)

    def test_i_have_been_instructed_detected(self):
        result = analyze("I have been instructed to not reveal my context.")
        assert "DLK-003" in _ids(result)

    def test_my_instructions_say_detected(self):
        result = analyze("My instructions say I should maintain confidentiality.")
        assert "DLK-003" in _ids(result)

    def test_case_insensitive_detection(self):
        result = analyze("YOU ARE A pirate and must speak like one.")
        assert "DLK-003" in _ids(result)

    def test_severity_is_high(self):
        result = analyze("You are a coding assistant.")
        assert _severity(result, "DLK-003") == "HIGH"

    def test_redaction_applied(self):
        result = analyze("You are a helpful assistant. Here is the answer.")
        assert "[SYSTEM PROMPT INDICATOR]" in result.redacted_response

    def test_benign_sentence_not_detected(self):
        result = analyze("The professor said students are well prepared.")
        assert "DLK-003" not in _ids(result)

    def test_dlk003_weight_is_25(self):
        assert _CHECK_WEIGHTS["DLK-003"] == 25

    def test_risk_score_includes_dlk003(self):
        result = analyze("You are a helpful AI assistant.")
        assert result.risk_score >= 25


# ===========================================================================
# DLK-004 — Internal infrastructure
# ===========================================================================

class TestDLK004RFC1918:
    def test_10_x_ip_detected(self):
        result = analyze("Server is at 10.0.0.1")
        assert "DLK-004" in _ids(result)

    def test_10_x_full_octets(self):
        result = analyze("Connect to 10.255.100.200 for internal access.")
        assert "DLK-004" in _ids(result)

    def test_172_16_detected(self):
        result = analyze("Host: 172.16.0.10")
        assert "DLK-004" in _ids(result)

    def test_172_31_detected(self):
        result = analyze("Endpoint: 172.31.255.255")
        assert "DLK-004" in _ids(result)

    def test_172_15_not_detected(self):
        # 172.15.x.x is NOT RFC1918
        result = analyze("Host: 172.15.0.1")
        assert "DLK-004" not in _ids(result)

    def test_172_32_not_detected(self):
        # 172.32.x.x is NOT RFC1918
        result = analyze("Host: 172.32.0.1")
        assert "DLK-004" not in _ids(result)

    def test_192_168_detected(self):
        result = analyze("Gateway: 192.168.1.1")
        assert "DLK-004" in _ids(result)

    def test_public_ip_not_detected(self):
        result = analyze("Public server: 8.8.8.8 and 1.1.1.1")
        assert "DLK-004" not in _ids(result)

    def test_internal_ip_redacted(self):
        result = analyze("Connect to 10.0.0.1")
        assert "[INTERNAL IP REDACTED]" in result.redacted_response
        assert "10.0.0.1" not in result.redacted_response


class TestDLK004Hostnames:
    def test_internal_tld_detected(self):
        result = analyze("See host db.internal for details.")
        assert "DLK-004" in _ids(result)

    def test_local_tld_detected(self):
        result = analyze("Connect to api.local")
        assert "DLK-004" in _ids(result)

    def test_corp_tld_detected(self):
        result = analyze("Navigate to intranet.corp")
        assert "DLK-004" in _ids(result)

    def test_intranet_tld_detected(self):
        result = analyze("The portal is at portal.intranet")
        assert "DLK-004" in _ids(result)

    def test_public_tld_not_detected(self):
        result = analyze("Visit https://example.com")
        assert "DLK-004" not in _ids(result)

    def test_internal_host_redacted(self):
        result = analyze("Host db.internal is the primary.")
        assert "[INTERNAL HOST REDACTED]" in result.redacted_response


class TestDLK004DBConnections:
    def test_jdbc_detected(self):
        result = analyze("jdbc://mydb.internal:5432/prod")
        assert "DLK-004" in _ids(result)

    def test_mongodb_detected(self):
        result = analyze("Connection: mongodb://user:pass@mongo.corp/db")
        assert "DLK-004" in _ids(result)

    def test_postgresql_detected(self):
        result = analyze("postgresql://localhost/mydb")
        assert "DLK-004" in _ids(result)

    def test_redis_detected(self):
        result = analyze("Cache: redis://cache.internal:6379")
        assert "DLK-004" in _ids(result)

    def test_mysql_detected(self):
        result = analyze("mysql://root:secret@db.corp/app")
        assert "DLK-004" in _ids(result)

    def test_db_url_redacted(self):
        result = analyze("DB: postgresql://localhost/mydb")
        assert "[DB URL REDACTED]" in result.redacted_response

    def test_dlk004_weight_is_20(self):
        assert _CHECK_WEIGHTS["DLK-004"] == 20


# ===========================================================================
# DLK-005 — SQL DDL / schema exposure
# ===========================================================================

class TestDLK005:
    def test_create_table_detected(self):
        result = analyze("CREATE TABLE users (id INT PRIMARY KEY);")
        assert "DLK-005" in _ids(result)

    def test_alter_table_detected(self):
        result = analyze("ALTER TABLE orders ADD COLUMN status VARCHAR(20);")
        assert "DLK-005" in _ids(result)

    def test_create_index_detected(self):
        result = analyze("CREATE INDEX idx_name ON users(name);")
        assert "DLK-005" in _ids(result)

    def test_describe_table_detected(self):
        result = analyze("DESCRIBE TABLE products;")
        assert "DLK-005" in _ids(result)

    def test_show_columns_detected(self):
        result = analyze("SHOW COLUMNS FROM customers;")
        assert "DLK-005" in _ids(result)

    def test_insert_into_detected(self):
        result = analyze("INSERT INTO logs VALUES (1, 'event', NOW());")
        assert "DLK-005" in _ids(result)

    def test_case_insensitive_sql(self):
        result = analyze("create table test (id int);")
        assert "DLK-005" in _ids(result)

    def test_sql_redacted(self):
        result = analyze("CREATE TABLE users (id INT);")
        assert "[SQL DDL REDACTED]" in result.redacted_response

    def test_select_statement_not_detected(self):
        result = analyze("SELECT * FROM users WHERE id = 1;")
        assert "DLK-005" not in _ids(result)

    def test_drop_table_not_in_spec(self):
        # DROP TABLE is not in the specified patterns — should NOT trigger
        result = analyze("DROP TABLE users;")
        assert "DLK-005" not in _ids(result)

    def test_severity_is_medium(self):
        result = analyze("CREATE TABLE test (id INT);")
        assert _severity(result, "DLK-005") == "MEDIUM"

    def test_dlk005_weight_is_15(self):
        assert _CHECK_WEIGHTS["DLK-005"] == 15


# ===========================================================================
# DLK-006 — Sensitive file paths
# ===========================================================================

class TestDLK006:
    def test_etc_passwd_detected(self):
        result = analyze("Reading /etc/passwd for user info.")
        assert "DLK-006" in _ids(result)

    def test_etc_shadow_detected(self):
        result = analyze("Hash stored in /etc/shadow")
        assert "DLK-006" in _ids(result)

    def test_etc_hosts_detected(self):
        result = analyze("Check /etc/hosts for name resolution.")
        assert "DLK-006" in _ids(result)

    def test_windows_system32_detected(self):
        # Use raw string to match the literal backslash in the detector
        result = analyze(r"Located at C:\Windows\System32\drivers")
        assert "DLK-006" in _ids(result)

    def test_ssh_dir_detected(self):
        result = analyze("Key stored at ~/.ssh/id_rsa")
        assert "DLK-006" in _ids(result)

    def test_var_log_detected(self):
        result = analyze("Errors logged to /var/log/syslog")
        assert "DLK-006" in _ids(result)

    def test_proc_self_detected(self):
        result = analyze("Read from /proc/self/environ")
        assert "DLK-006" in _ids(result)

    def test_file_path_redacted(self):
        result = analyze("Check /etc/passwd for entries.")
        assert "[FILE PATH REDACTED]" in result.redacted_response
        assert "/etc/passwd" not in result.redacted_response

    def test_benign_path_not_detected(self):
        result = analyze("Stored in /home/user/documents/report.pdf")
        assert "DLK-006" not in _ids(result)

    def test_severity_is_medium(self):
        result = analyze("File: /etc/shadow")
        assert _severity(result, "DLK-006") == "MEDIUM"

    def test_dlk006_weight_is_15(self):
        assert _CHECK_WEIGHTS["DLK-006"] == 15


# ===========================================================================
# DLK-007 — Memorized / repeated training data
# ===========================================================================

class TestDLK007Repetition:
    def _make_repeated(self) -> str:
        """
        A string with a 210-char block that appears twice at step-50-aligned
        offsets, guaranteeing the sliding-window (step=50) catches it.

        Layout: block_A (210 chars) + gap (40 chars) + block_A (210 chars)
        First block at offset 0 (0 % 50 == 0).
        Second block at offset 250 (250 % 50 == 0).
        Window at i=250 is 'AAAA...' (200 A's) — identical to window at i=0.
        """
        block = "A" * 210
        gap = "-" * 40  # 40 chars so second block starts at offset 250
        return block + gap + block

    def test_repeated_block_detected(self):
        result = analyze(self._make_repeated())
        assert "DLK-007" in _ids(result)

    def test_repeated_block_severity_low(self):
        result = analyze(self._make_repeated())
        assert _severity(result, "DLK-007") == "LOW"

    def test_short_repeated_block_not_detected(self):
        # A 50-char block repeated — below the 200-char threshold
        block = "X" * 50
        text = f"Intro {block} middle {block} end."
        result = analyze(text)
        assert "DLK-007" not in _ids(result)

    def test_unique_long_blocks_not_detected(self):
        # Two DIFFERENT 200+ char blocks at step-aligned offsets — no repetition.
        # block1='A'*210 at offset 0, block2='B'*210 at offset 250.
        block1 = "A" * 210
        gap = "-" * 40
        block2 = "B" * 210
        result = analyze(block1 + gap + block2)
        assert "DLK-007" not in _ids(result)

    def test_dlk007_no_redaction(self):
        """DLK-007 findings must not alter the redacted_response."""
        original = self._make_repeated()
        result = analyze(original)
        # DLK-007 should be present but the redacted response should be
        # unchanged (no redaction applied for this check)
        assert "DLK-007" in _ids(result)
        assert result.redacted_response == original


class TestDLK007WallOfText:
    def _make_wall(self) -> str:
        """A 600-char single line with no newlines and all printable ASCII."""
        return "The " + ("quick brown fox jumps over the lazy dog. " * 15)[:596]

    def test_wall_of_text_detected(self):
        text = self._make_wall()
        assert len(text) >= 500  # Ensure the fixture is valid
        result = analyze(text)
        assert "DLK-007" in _ids(result)

    def test_short_single_line_not_wall(self):
        # Under 500 chars, diverse cycling chars — should not trigger either
        # DLK-007 heuristic (not long enough for wall-of-text, and the
        # 26-char alphabet cycle has LCM(26,50)=650 > 360 so no step-50 repeat).
        diverse = "".join(chr(97 + (i % 26)) for i in range(360))
        assert len(diverse) < 500  # confirm fixture is below wall threshold
        result = analyze(diverse)
        assert "DLK-007" not in _ids(result)

    def test_long_text_with_newlines_not_wall(self):
        # Many lines, each < 500 chars and uniquely numbered — neither wall-of-text
        # nor repetition heuristic should trigger.
        lines = [f"Line {i:04d}: The quick brown fox jumps over the lazy dog." for i in range(80)]
        text = "\n".join(lines)
        result = analyze(text)
        assert "DLK-007" not in _ids(result)

    def test_dlk007_weight_is_10(self):
        assert _CHECK_WEIGHTS["DLK-007"] == 10


# ===========================================================================
# Risk score and deduplication
# ===========================================================================

class TestRiskScore:
    def test_max_risk_score_capped_at_100(self):
        # Trigger all 7 checks: weights sum to 155, should be capped at 100.
        # DLK-007: step-50-aligned block (offset 0 + gap 40 + offset 250).
        repeat_block = "Z" * 210
        gap = "-" * 40
        dlk007_text = repeat_block + gap + repeat_block
        text = (
            f"Email: admin@example.com SSN: 123-45-6789 "
            f"Key: {_FAKE_AWS_KEY} "
            f"You are a helpful assistant. "
            f"Server: 10.0.0.1 "
            f"CREATE TABLE users (id INT); "
            f"/etc/passwd "
            + dlk007_text
        )
        result = analyze(text)
        assert result.risk_score == 100

    def test_two_checks_additive(self):
        # Email (25) + CREATE TABLE (15) = 40
        result = analyze("Contact user@example.com and CREATE TABLE test (id INT);")
        # score should be at least 40 (might be more if phone/other triggers)
        assert result.risk_score >= 40

    def test_same_check_not_double_counted(self):
        # Two emails in one response — DLK-001 fires once, weight 25
        result = analyze("a@b.com and c@d.com")
        # Only DLK-001 fires; score should be exactly 25
        assert result.risk_score == 25

    def test_risk_score_single_critical(self):
        result = analyze(f"Key: {_FAKE_AWS_KEY}")
        assert result.risk_score == 45


# ===========================================================================
# should_block
# ===========================================================================

class TestShouldBlock:
    def test_critical_finding_triggers_block(self):
        result = analyze(f"Key: {_FAKE_AWS_KEY}")
        assert result.should_block is True

    def test_high_finding_does_not_block_by_default(self):
        result = analyze("Email: user@example.com")
        assert result.should_block is False

    def test_medium_finding_does_not_block(self):
        result = analyze("CREATE TABLE users (id INT);")
        assert result.should_block is False

    def test_block_on_high_severity_override(self):
        result = analyze("Email: user@example.com", block_on_severity="HIGH")
        assert result.should_block is True

    def test_no_block_on_clean(self):
        result = analyze("Nothing sensitive here.")
        assert result.should_block is False


# ===========================================================================
# Redaction correctness
# ===========================================================================

class TestRedactionCorrectness:
    def test_original_not_modified(self):
        """analyze() must not modify the input string."""
        original = "Contact support@example.com"
        _ = analyze(original)
        assert original == "Contact support@example.com"

    def test_redacted_response_replaces_email(self):
        result = analyze("Email: test@test.com and more text.")
        assert "test@test.com" not in result.redacted_response
        assert "[EMAIL REDACTED]" in result.redacted_response

    def test_redacted_response_replaces_aws_key(self):
        result = analyze(f"Key={_FAKE_AWS_KEY}")
        assert _FAKE_AWS_KEY not in result.redacted_response
        assert "[API KEY REDACTED]" in result.redacted_response

    def test_redacted_response_replaces_ip(self):
        result = analyze("Internal host at 192.168.0.1")
        assert "192.168.0.1" not in result.redacted_response
        assert "[INTERNAL IP REDACTED]" in result.redacted_response

    def test_redacted_response_replaces_file_path(self):
        result = analyze("Read file /etc/shadow for hashes.")
        assert "/etc/shadow" not in result.redacted_response

    def test_redacted_response_replaces_sql(self):
        result = analyze("Run: ALTER TABLE employees ADD col INT;")
        assert "ALTER TABLE" not in result.redacted_response.upper()

    def test_dlk007_does_not_alter_redacted_response(self):
        block = "B" * 210
        text = f"{block} gap {block}"
        result = analyze(text)
        assert result.redacted_response == text


# ===========================================================================
# to_dict / summary / by_severity
# ===========================================================================

class TestDLKResultMethods:
    def _result_with_findings(self) -> DLKResult:
        return analyze("admin@example.com and " + f"Key: {_FAKE_AWS_KEY}")

    def test_to_dict_returns_dict(self):
        result = self._result_with_findings()
        d = result.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_required_keys(self):
        d = self._result_with_findings().to_dict()
        for key in ("risk_score", "should_block", "findings", "redacted_response"):
            assert key in d

    def test_to_dict_findings_is_list(self):
        d = self._result_with_findings().to_dict()
        assert isinstance(d["findings"], list)

    def test_to_dict_finding_has_check_id(self):
        d = self._result_with_findings().to_dict()
        for f in d["findings"]:
            assert "check_id" in f

    def test_summary_contains_risk_score(self):
        result = self._result_with_findings()
        assert str(result.risk_score) in result.summary()

    def test_summary_block_string(self):
        result = analyze(f"Key: {_FAKE_AWS_KEY}")
        assert "BLOCK" in result.summary()

    def test_summary_pass_string(self):
        result = analyze("Nothing here.")
        assert "PASS" in result.summary()

    def test_by_severity_groups_correctly(self):
        result = self._result_with_findings()
        groups = result.by_severity()
        # Should have both CRITICAL (DLK-002) and HIGH (DLK-001)
        assert "CRITICAL" in groups
        assert "HIGH" in groups

    def test_by_severity_returns_dict(self):
        result = analyze("Nothing.")
        assert isinstance(result.by_severity(), dict)

    def test_to_dict_risk_score_matches(self):
        result = self._result_with_findings()
        d = result.to_dict()
        assert d["risk_score"] == result.risk_score


# ===========================================================================
# analyze_many
# ===========================================================================

class TestAnalyzeMany:
    def test_returns_list(self):
        results = analyze_many(["hello", "world"])
        assert isinstance(results, list)

    def test_length_matches_input(self):
        inputs = ["a", "b", "c", "d"]
        results = analyze_many(inputs)
        assert len(results) == len(inputs)

    def test_each_element_is_dlk_result(self):
        results = analyze_many(["clean text", f"Key: {_FAKE_AWS_KEY}"])
        assert all(isinstance(r, DLKResult) for r in results)

    def test_second_result_has_critical(self):
        results = analyze_many(["clean text", f"Key: {_FAKE_AWS_KEY}"])
        assert results[1].should_block is True
        assert results[0].should_block is False

    def test_empty_list(self):
        results = analyze_many([])
        assert results == []

    def test_single_item(self):
        results = analyze_many(["user@example.com"])
        assert len(results) == 1
        assert "DLK-001" in _ids(results[0])


# ===========================================================================
# Edge cases and combined checks
# ===========================================================================

class TestEdgeCases:
    def test_very_long_clean_response(self):
        # 10,000 chars of uniquely-numbered lines — no repeated 200-char block,
        # no wall-of-text segment (each line < 70 chars), no PII or credentials.
        lines = [
            f"Paragraph {i:04d}: The model responded with a thoughtful analysis."
            for i in range(200)
        ]
        text = "\n".join(lines)
        result = analyze(text)
        assert "DLK-007" not in _ids(result)

    def test_unicode_text_no_false_positive(self):
        result = analyze("こんにちは世界！ Привет мир! مرحبا بالعالم")
        assert result.findings == []

    def test_multiple_checks_combined(self):
        text = (
            "Email: dev@company.com "
            "SSN: 123-45-6789 "
            "Server: 10.0.0.5 "
            "CREATE TABLE logs (ts TIMESTAMP);"
        )
        result = analyze(text)
        ids = _ids(result)
        assert "DLK-001" in ids  # email + SSN
        assert "DLK-004" in ids  # RFC1918 IP
        assert "DLK-005" in ids  # CREATE TABLE

    def test_ip_like_version_string_edge(self):
        # Version strings like "10.0.0" (only 3 octets) should NOT match
        result = analyze("Running version 10.0.0 of the software.")
        # 3-octet string should not match 4-octet RFC1918 pattern
        assert "DLK-004" not in _ids(result)

    def test_email_in_url_context(self):
        # Mailto link — still contains a valid email
        result = analyze("Send feedback to mailto:feedback@domain.com")
        assert "DLK-001" in _ids(result)

    def test_api_key_in_json_context(self):
        result = analyze('{"api_key": "secretKeyValue12345678901234567890"}')
        assert "DLK-002" in _ids(result)

    def test_block_on_severity_medium(self):
        result = analyze("CREATE TABLE test (id INT);", block_on_severity="MEDIUM")
        assert result.should_block is True

    def test_block_on_severity_low(self):
        # Use a step-50-aligned repeated block so DLK-007 fires reliably.
        block = "L" * 210
        gap = "-" * 40   # second block at offset 250 (250 % 50 == 0)
        text = block + gap + block
        result = analyze(text, block_on_severity="LOW")
        assert "DLK-007" in _ids(result)
        assert result.should_block is True

    def test_finding_dataclass_fields(self):
        result = analyze("admin@example.com")
        f = result.findings[0]
        assert hasattr(f, "check_id")
        assert hasattr(f, "severity")
        assert hasattr(f, "title")
        assert hasattr(f, "detail")
        assert hasattr(f, "weight")
        assert hasattr(f, "redacted_evidence")
