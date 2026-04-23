from guardrails.output_filter import OutputFilter


def test_redacts_aws_access_key_like_token():
    f = OutputFilter(policy={"secret_output_action": "redact"})
    text = "Here is the key AKIA1234567890ABCDEF do not share."
    result = f.filter(text)

    assert result.action == "redact"
    assert "[REDACTED_SECRET]" in result.content
    assert "AKIA1234567890ABCDEF" not in result.content


def test_blocks_jwt_when_policy_block():
    f = OutputFilter(policy={"secret_output_action": "block"})
    jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4iLCJpYXQiOjE1MTYyMzkwMjJ9.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    result = f.filter(f"token: {jwt}")

    assert result.action == "block"
    assert "BLOCKED" in result.content


def test_redacts_high_entropy_long_token():
    f = OutputFilter(policy={"secret_output_action": "redact"})
    token = "QWxhZGRpbjpPcGVuU2VzYW1lMTIzNDU2Nzg5MEFCQ0RFRkdISUo="
    result = f.filter(f"leaked={token}")

    assert result.action == "redact"
    assert "[REDACTED_SECRET]" in result.content


def test_does_not_flag_normal_text_false_positive_safe():
    f = OutputFilter(policy={"secret_output_action": "redact"})
    text = "The meeting is scheduled for Tuesday at 10am in Room 204."
    result = f.filter(text)

    assert result.action == "allow"
    assert result.content == text
