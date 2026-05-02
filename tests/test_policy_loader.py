import hashlib

from guardrails.policy_loader import load_policy


def test_load_policy_exposes_startup_sha256_and_logs(caplog, tmp_path):
    policy_file = tmp_path / "policy.yaml"
    raw = b"version: 1\nrules:\n  - id: test\n"
    policy_file.write_bytes(raw)

    with caplog.at_level("INFO"):
        policy, runtime = load_policy(policy_file)

    assert policy["version"] == 1
    assert runtime.policy_file_path == str(policy_file)
    assert runtime.policy_file_sha256 == hashlib.sha256(raw).hexdigest()

    assert any(
        r.message == "Policy initialized"
        and getattr(r, "policy_file_sha256", None) == runtime.policy_file_sha256
        for r in caplog.records
    )
