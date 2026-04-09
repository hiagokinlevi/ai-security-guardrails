# test_action_auditor.py — Test suite for action_auditor module
# Part of Cyber Port portfolio: github.com/hiagokinlevi/k1N-ai-security-guardrails
#
# License: Creative Commons Attribution 4.0 International (CC BY 4.0)
# https://creativecommons.org/licenses/by/4.0/
#
# Run with: python -m pytest tests/test_action_auditor.py -q

import sys
import os

# Ensure the repo root is on sys.path so the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from guardrails.agent.action_auditor import (
    ToolCall,
    AGTFinding,
    AGTResult,
    audit,
    audit_sequence,
    _CHECK_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def make_call(
    tool_name: str,
    arguments: dict = None,
    call_id: str = "c1",
    timestamp_ms: int = 1000000,
    depth: int = 0,
) -> ToolCall:
    return ToolCall(
        call_id=call_id,
        tool_name=tool_name,
        arguments=arguments or {},
        timestamp_ms=timestamp_ms,
        depth=depth,
    )


def ids_in(result: AGTResult) -> set:
    """Return the set of check IDs present in a result."""
    return {f.check_id for f in result.findings}


# ===========================================================================
# AGT-001: Filesystem boundary violations
# ===========================================================================

class TestAGT001:

    # --- allowlist mode ---

    def test_001_allowlist_path_inside_allowed(self):
        """Path under an allowed dir must not fire."""
        call = make_call("read_file", {"path": "/workspace/project/main.py"})
        result = audit(call, allowed_working_dirs=["/workspace"])
        assert "AGT-001" not in ids_in(result)

    def test_001_allowlist_path_outside_allowed(self):
        """Path outside allowed dir must fire AGT-001."""
        call = make_call("read_file", {"path": "/etc/passwd"})
        result = audit(call, allowed_working_dirs=["/workspace"])
        assert "AGT-001" in ids_in(result)

    def test_001_allowlist_path_prefix_match(self):
        """Allowed dir must be a prefix — a sibling dir must fire."""
        call = make_call("write_file", {"path": "/workspace2/secret.txt"})
        result = audit(call, allowed_working_dirs=["/workspace"])
        assert "AGT-001" in ids_in(result)

    def test_001_allowlist_multiple_dirs_allowed(self):
        """Path matching second allowed dir must not fire."""
        call = make_call("open_file", {"path": "/data/inputs/file.csv"})
        result = audit(call, allowed_working_dirs=["/workspace", "/data"])
        assert "AGT-001" not in ids_in(result)

    def test_001_allowlist_multiple_dirs_outside(self):
        """Path matching neither of two allowed dirs must fire."""
        call = make_call("read_file", {"path": "/home/user/private"})
        result = audit(call, allowed_working_dirs=["/workspace", "/data"])
        assert "AGT-001" in ids_in(result)

    def test_001_allowlist_tilde_path_outside(self):
        """Tilde path outside allowlist must fire."""
        call = make_call("read_file", {"path": "~/secret"})
        result = audit(call, allowed_working_dirs=["/workspace"])
        assert "AGT-001" in ids_in(result)

    def test_001_allowlist_relative_traversal_outside(self):
        """Relative traversal path outside allowlist must fire."""
        call = make_call("read_file", {"path": "../config/settings.cfg"})
        result = audit(call, allowed_working_dirs=["/workspace"])
        assert "AGT-001" in ids_in(result)

    # --- no-allowlist mode ---

    def test_001_no_allowlist_traversal_fires(self):
        """Path with .. must fire when no allowlist is set."""
        call = make_call("read_file", {"path": "../../etc/passwd"})
        result = audit(call)
        assert "AGT-001" in ids_in(result)

    def test_001_no_allowlist_etc_fires(self):
        """/etc path must fire when no allowlist is set."""
        call = make_call("cat", {"path": "/etc/shadow"})
        result = audit(call)
        assert "AGT-001" in ids_in(result)

    def test_001_no_allowlist_var_fires(self):
        """/var path must fire when no allowlist is set."""
        call = make_call("read_file", {"path": "/var/log/syslog"})
        result = audit(call)
        assert "AGT-001" in ids_in(result)

    def test_001_no_allowlist_proc_fires(self):
        """/proc path must fire when no allowlist is set."""
        call = make_call("read_file", {"path": "/proc/self/environ"})
        result = audit(call)
        assert "AGT-001" in ids_in(result)

    def test_001_no_allowlist_sys_fires(self):
        """/sys path must fire when no allowlist is set."""
        call = make_call("read", {"path": "/sys/kernel/debug"})
        result = audit(call)
        assert "AGT-001" in ids_in(result)

    def test_001_no_allowlist_normal_path_clean(self):
        """A non-sensitive path without traversal must not fire."""
        call = make_call("read_file", {"path": "/home/user/documents/report.pdf"})
        result = audit(call)
        assert "AGT-001" not in ids_in(result)

    def test_001_non_fs_tool_ignored(self):
        """A non-filesystem tool must not trigger AGT-001."""
        call = make_call("send_email", {"path": "/etc/passwd"})
        result = audit(call)
        assert "AGT-001" not in ids_in(result)

    def test_001_non_path_arg_ignored(self):
        """An argument value that is not a path must not trigger AGT-001."""
        call = make_call("read_file", {"content": "hello world"})
        result = audit(call)
        assert "AGT-001" not in ids_in(result)

    def test_001_severity_is_critical(self):
        """AGT-001 finding must carry CRITICAL severity."""
        call = make_call("read_file", {"path": "/etc/passwd"})
        result = audit(call)
        finding = next(f for f in result.findings if f.check_id == "AGT-001")
        assert finding.severity == "CRITICAL"

    def test_001_weight_is_45(self):
        """AGT-001 weight must be 45."""
        assert _CHECK_WEIGHTS["AGT-001"] == 45

    def test_001_windows_path_outside_allowlist(self):
        """A Windows absolute path outside the allowlist must fire."""
        call = make_call("read_file", {"path": r"C:\Users\admin\secret.txt"})
        result = audit(call, allowed_working_dirs=["/workspace"])
        assert "AGT-001" in ids_in(result)

    def test_001_allowlist_exact_match(self):
        """A path exactly equal to an allowed dir must not fire."""
        call = make_call("list_dir", {"path": "/workspace"})
        result = audit(call, allowed_working_dirs=["/workspace"])
        assert "AGT-001" not in ids_in(result)

    def test_001_mkdir_tool_triggers_check(self):
        """mkdir tool name must activate AGT-001 check."""
        call = make_call("mkdir", {"path": "/etc/newdir"})
        result = audit(call)
        assert "AGT-001" in ids_in(result)

    def test_001_delete_tool_triggers_check(self):
        """delete tool name must activate AGT-001 check."""
        call = make_call("delete_file", {"path": "/var/important"})
        result = audit(call)
        assert "AGT-001" in ids_in(result)


# ===========================================================================
# AGT-002: Burst rate abuse
# ===========================================================================

class TestAGT002:

    def _make_history(self, n: int, base_ts: int = 900000) -> list:
        """Create n ToolCall objects spaced 1 ms apart within the window."""
        return [
            make_call("any_tool", call_id=f"h{i}", timestamp_ms=base_ts + i)
            for i in range(n)
        ]

    def test_002_exactly_at_limit_no_fire(self):
        """Exactly burst_limit calls (history + current) must not fire."""
        # burst_limit=10, so 9 history + 1 current = 10 (not over)
        history = self._make_history(9)
        call = make_call("any_tool", timestamp_ms=1000000)
        result = audit(call, call_history=history, burst_limit=10, burst_window_ms=200000)
        assert "AGT-002" not in ids_in(result)

    def test_002_one_over_limit_fires(self):
        """burst_limit + 1 calls must fire AGT-002."""
        # burst_limit=10, so 10 history + 1 current = 11
        history = self._make_history(10)
        call = make_call("any_tool", timestamp_ms=1000000)
        result = audit(call, call_history=history, burst_limit=10, burst_window_ms=200000)
        assert "AGT-002" in ids_in(result)

    def test_002_calls_outside_window_not_counted(self):
        """Calls older than burst_window_ms must not count toward burst."""
        # All history calls are 200001 ms before current — outside the 200000 ms window
        old_history = self._make_history(20, base_ts=100000)
        call = make_call("any_tool", timestamp_ms=1000000)
        result = audit(call, call_history=old_history, burst_limit=10, burst_window_ms=200000)
        assert "AGT-002" not in ids_in(result)

    def test_002_mixed_window_counts_correctly(self):
        """Only history calls within the window should count."""
        # 5 old calls outside window + 8 recent calls inside => 9 total (8+1), under limit 10
        old = self._make_history(5, base_ts=100000)
        recent = self._make_history(8, base_ts=950000)
        call = make_call("any_tool", timestamp_ms=1000000)
        result = audit(call, call_history=old + recent, burst_limit=10, burst_window_ms=200000)
        assert "AGT-002" not in ids_in(result)

    def test_002_mixed_window_over_limit(self):
        """Recent calls inside window that exceed limit must fire."""
        # 10 recent history + 1 current = 11 > 10
        old = self._make_history(5, base_ts=100000)
        recent = self._make_history(10, base_ts=950000)
        call = make_call("any_tool", timestamp_ms=1000000)
        result = audit(call, call_history=old + recent, burst_limit=10, burst_window_ms=200000)
        assert "AGT-002" in ids_in(result)

    def test_002_empty_history_no_fire(self):
        """No history with burst_limit=1 still needs >1 to fire."""
        call = make_call("any_tool", timestamp_ms=1000000)
        result = audit(call, call_history=[], burst_limit=1, burst_window_ms=60000)
        assert "AGT-002" not in ids_in(result)

    def test_002_severity_is_high(self):
        """AGT-002 finding must carry HIGH severity."""
        history = self._make_history(10)
        call = make_call("any_tool", timestamp_ms=1000000)
        result = audit(call, call_history=history, burst_limit=10, burst_window_ms=200000)
        finding = next(f for f in result.findings if f.check_id == "AGT-002")
        assert finding.severity == "HIGH"

    def test_002_weight_is_25(self):
        """AGT-002 weight must be 25."""
        assert _CHECK_WEIGHTS["AGT-002"] == 25

    def test_002_default_burst_limit_not_exceeded(self):
        """5 calls within the default window must not fire with default burst_limit=10."""
        history = self._make_history(4, base_ts=999000)
        call = make_call("any_tool", timestamp_ms=1000000)
        result = audit(call, call_history=history)
        assert "AGT-002" not in ids_in(result)

    def test_002_boundary_exactly_window_edge_included(self):
        """A call exactly at window start (timestamp_ms - burst_window_ms) is included."""
        window_ms = 60000
        base_ts = 1000000
        # history call is exactly at window start
        history = [make_call("any_tool", call_id="h1", timestamp_ms=base_ts - window_ms)]
        call = make_call("any_tool", timestamp_ms=base_ts)
        # 1 history + 1 current = 2, burst_limit=1 => should fire
        result = audit(call, call_history=history, burst_limit=1, burst_window_ms=window_ms)
        assert "AGT-002" in ids_in(result)


# ===========================================================================
# AGT-003: Recursive / stuck-loop pattern
# ===========================================================================

class TestAGT003:

    def _same_calls(self, n: int, tool: str = "search_tool", args: dict = None) -> list:
        args = args or {"query": "find me"}
        return [
            make_call(tool, args, call_id=f"h{i}", timestamp_ms=i * 100)
            for i in range(n)
        ]

    def test_003_three_identical_history_fires(self):
        """3 history + current all same tool + same args must fire AGT-003."""
        history = self._same_calls(3)
        call = make_call("search_tool", {"query": "find me"})
        result = audit(call, call_history=history)
        assert "AGT-003" in ids_in(result)

    def test_003_only_two_history_same_no_fire(self):
        """Only 2 history calls must not fire (need last 3 to all match)."""
        history = self._same_calls(2)
        call = make_call("search_tool", {"query": "find me"})
        result = audit(call, call_history=history)
        assert "AGT-003" not in ids_in(result)

    def test_003_different_tool_name_no_fire(self):
        """Consecutive calls with different tool names must not fire AGT-003."""
        history = [
            make_call("tool_a", {"q": "x"}, call_id="h1"),
            make_call("tool_b", {"q": "x"}, call_id="h2"),
            make_call("tool_a", {"q": "x"}, call_id="h3"),
        ]
        call = make_call("tool_a", {"q": "x"})
        result = audit(call, call_history=history)
        assert "AGT-003" not in ids_in(result)

    def test_003_all_different_args_no_fire(self):
        """Same tool but all different args — fewer than 2 identical — must not fire."""
        history = [
            make_call("search_tool", {"query": "alpha"}, call_id="h1"),
            make_call("search_tool", {"query": "beta"},  call_id="h2"),
            make_call("search_tool", {"query": "gamma"}, call_id="h3"),
        ]
        call = make_call("search_tool", {"query": "delta"})
        result = audit(call, call_history=history)
        assert "AGT-003" not in ids_in(result)

    def test_003_two_of_three_identical_fires(self):
        """Same tool, 2 of last 3 history have same args as current — must fire."""
        history = [
            make_call("search_tool", {"query": "find me"},  call_id="h1"),
            make_call("search_tool", {"query": "find me"},  call_id="h2"),
            make_call("search_tool", {"query": "different"}, call_id="h3"),
        ]
        call = make_call("search_tool", {"query": "find me"})
        result = audit(call, call_history=history)
        assert "AGT-003" in ids_in(result)

    def test_003_only_looks_at_last_three(self):
        """Identical calls deeper than 3 in history must not cause AGT-003 by themselves."""
        # Last 3 history have different args; repetition is only in older calls
        history = [
            make_call("search_tool", {"query": "find me"}, call_id=f"h{i}")
            for i in range(10)
        ] + [
            make_call("search_tool", {"query": "alpha"}, call_id="h10"),
            make_call("search_tool", {"query": "beta"},  call_id="h11"),
            make_call("search_tool", {"query": "gamma"}, call_id="h12"),
        ]
        call = make_call("search_tool", {"query": "find me"})
        result = audit(call, call_history=history)
        # Last 3 history: alpha/beta/gamma — not identical to current "find me"
        # Only 0 of last 3 match current args, so should NOT fire
        assert "AGT-003" not in ids_in(result)

    def test_003_severity_is_high(self):
        """AGT-003 finding must carry HIGH severity."""
        history = self._same_calls(3)
        call = make_call("search_tool", {"query": "find me"})
        result = audit(call, call_history=history)
        finding = next(f for f in result.findings if f.check_id == "AGT-003")
        assert finding.severity == "HIGH"

    def test_003_weight_is_25(self):
        """AGT-003 weight must be 25."""
        assert _CHECK_WEIGHTS["AGT-003"] == 25

    def test_003_empty_args_repetition_fires(self):
        """Empty args repeated counts as identical and must fire."""
        history = [make_call("ping_tool", {}, call_id=f"h{i}") for i in range(3)]
        call = make_call("ping_tool", {})
        result = audit(call, call_history=history)
        assert "AGT-003" in ids_in(result)

    def test_003_no_history_no_fire(self):
        """No call history must not fire AGT-003."""
        call = make_call("search_tool", {"query": "find me"})
        result = audit(call, call_history=[])
        assert "AGT-003" not in ids_in(result)


# ===========================================================================
# AGT-004: Unauthorized network domain
# ===========================================================================

class TestAGT004:

    def test_004_allowed_domain_no_fire(self):
        """A request to an allowed domain must not fire AGT-004."""
        call = make_call("http_get", {"url": "https://api.example.com/data"})
        result = audit(call, allowed_domains=["api.example.com"])
        assert "AGT-004" not in ids_in(result)

    def test_004_unauthorized_domain_fires(self):
        """A request to an unlisted domain must fire AGT-004."""
        call = make_call("http_get", {"url": "https://evil.attacker.com/exfil"})
        result = audit(call, allowed_domains=["api.example.com"])
        assert "AGT-004" in ids_in(result)

    def test_004_no_allowed_domains_no_fire(self):
        """When allowed_domains is empty/None, AGT-004 must not fire."""
        call = make_call("http_get", {"url": "https://any.domain.com/data"})
        result = audit(call, allowed_domains=[])
        assert "AGT-004" not in ids_in(result)

    def test_004_no_allowed_domains_none_no_fire(self):
        """When allowed_domains is None, AGT-004 must not fire."""
        call = make_call("fetch", {"url": "https://any.domain.com/data"})
        result = audit(call, allowed_domains=None)
        assert "AGT-004" not in ids_in(result)

    def test_004_non_network_tool_no_fire(self):
        """A non-network tool with a URL argument must not fire AGT-004."""
        call = make_call("read_file", {"path": "https://evil.com/file"})
        result = audit(call, allowed_domains=["safe.com"])
        assert "AGT-004" not in ids_in(result)

    def test_004_fetch_tool_fires(self):
        """fetch tool with unauthorized domain must fire."""
        call = make_call("fetch", {"url": "https://exfil.bad.org/data"})
        result = audit(call, allowed_domains=["good.com"])
        assert "AGT-004" in ids_in(result)

    def test_004_browse_tool_fires(self):
        """browse tool with unauthorized domain must fire."""
        call = make_call("browse", {"url": "https://not-allowed.net/"})
        result = audit(call, allowed_domains=["allowed.net"])
        assert "AGT-004" in ids_in(result)

    def test_004_curl_tool_fires(self):
        """curl tool with unauthorized domain must fire."""
        call = make_call("curl", {"url": "https://bad.example.org/"})
        result = audit(call, allowed_domains=["good.example.org"])
        assert "AGT-004" in ids_in(result)

    def test_004_no_url_in_args_no_fire(self):
        """A network tool with no URL in arguments must not fire AGT-004."""
        call = make_call("http_get", {"headers": "Accept: application/json"})
        result = audit(call, allowed_domains=["allowed.com"])
        assert "AGT-004" not in ids_in(result)

    def test_004_severity_is_high(self):
        """AGT-004 finding must carry HIGH severity."""
        call = make_call("http_get", {"url": "https://evil.com/steal"})
        result = audit(call, allowed_domains=["safe.com"])
        finding = next(f for f in result.findings if f.check_id == "AGT-004")
        assert finding.severity == "HIGH"

    def test_004_weight_is_25(self):
        """AGT-004 weight must be 25."""
        assert _CHECK_WEIGHTS["AGT-004"] == 25

    def test_004_http_url_fires(self):
        """http (not https) URL to unauthorized domain must also fire."""
        call = make_call("http_get", {"url": "http://evil.org/data"})
        result = audit(call, allowed_domains=["safe.com"])
        assert "AGT-004" in ids_in(result)

    def test_004_multiple_urls_first_bad_fires(self):
        """If the first URL extracted is unauthorized, AGT-004 fires."""
        call = make_call("http_get", {"url": "https://evil.com/a https://safe.com/b"})
        result = audit(call, allowed_domains=["safe.com"])
        assert "AGT-004" in ids_in(result)

    def test_004_web_search_tool_fires(self):
        """web_search tool with unauthorized domain must fire."""
        call = make_call("web_search", {"query": "https://evil.com search"})
        result = audit(call, allowed_domains=["good.com"])
        assert "AGT-004" in ids_in(result)


# ===========================================================================
# AGT-005: Code execution tool invoked without confirmation
# ===========================================================================

class TestAGT005:

    def test_005_execute_tool_with_command_fires(self):
        """execute tool with a non-empty command must fire AGT-005."""
        call = make_call("execute_command", {"cmd": "rm -rf /"})
        result = audit(call)
        assert "AGT-005" in ids_in(result)

    def test_005_bash_tool_fires(self):
        """bash tool with a non-empty command must fire AGT-005."""
        call = make_call("bash", {"script": "cat /etc/passwd"})
        result = audit(call)
        assert "AGT-005" in ids_in(result)

    def test_005_shell_tool_fires(self):
        """shell tool with a non-empty command must fire AGT-005."""
        call = make_call("run_shell", {"cmd": "ls -la"})
        result = audit(call)
        assert "AGT-005" in ids_in(result)

    def test_005_eval_tool_fires(self):
        """eval tool with a non-empty expression must fire AGT-005."""
        call = make_call("eval_expression", {"expr": "__import__('os').system('id')"})
        result = audit(call)
        assert "AGT-005" in ids_in(result)

    def test_005_subprocess_tool_fires(self):
        """subprocess tool with a non-empty command must fire AGT-005."""
        call = make_call("subprocess_run", {"args": "ping localhost"})
        result = audit(call)
        assert "AGT-005" in ids_in(result)

    def test_005_python_repl_tool_fires(self):
        """python_repl tool with non-empty code must fire AGT-005."""
        call = make_call("python_repl", {"code": "import os; os.system('id')"})
        result = audit(call)
        assert "AGT-005" in ids_in(result)

    def test_005_node_tool_fires(self):
        """node tool with non-empty code must fire AGT-005."""
        call = make_call("node_runner", {"code": "require('child_process').exec('id')"})
        result = audit(call)
        assert "AGT-005" in ids_in(result)

    def test_005_run_script_fires(self):
        """run_script tool with non-empty script must fire AGT-005."""
        call = make_call("run_script", {"script": "#!/bin/bash\necho hello"})
        result = audit(call)
        assert "AGT-005" in ids_in(result)

    def test_005_run_code_fires(self):
        """run_code tool with non-empty code must fire AGT-005."""
        call = make_call("run_code", {"code": "print('hello')"})
        result = audit(call)
        assert "AGT-005" in ids_in(result)

    def test_005_empty_args_no_fire(self):
        """Code execution tool with all-empty argument values must not fire AGT-005."""
        call = make_call("execute_command", {"cmd": ""})
        result = audit(call)
        assert "AGT-005" not in ids_in(result)

    def test_005_safe_tool_no_fire(self):
        """A non-execution tool must not fire AGT-005."""
        call = make_call("search_tool", {"query": "find execute patterns"})
        result = audit(call)
        assert "AGT-005" not in ids_in(result)

    def test_005_case_insensitive_match(self):
        """Tool name check must be case-insensitive."""
        call = make_call("Execute_Command", {"cmd": "whoami"})
        result = audit(call)
        assert "AGT-005" in ids_in(result)

    def test_005_severity_is_critical(self):
        """AGT-005 finding must carry CRITICAL severity."""
        call = make_call("bash", {"script": "id"})
        result = audit(call)
        finding = next(f for f in result.findings if f.check_id == "AGT-005")
        assert finding.severity == "CRITICAL"

    def test_005_weight_is_40(self):
        """AGT-005 weight must be 40."""
        assert _CHECK_WEIGHTS["AGT-005"] == 40

    def test_005_no_args_no_fire(self):
        """Code execution tool with no arguments at all must not fire."""
        call = make_call("bash", {})
        result = audit(call)
        assert "AGT-005" not in ids_in(result)


# ===========================================================================
# AGT-006: Sensitive data (PII / credentials) in arguments
# ===========================================================================

# Fake key constructed dynamically, matching the module's dynamic pattern
_FAKE_API_KEY = "AKIA" + "TESTKEY1234567890"[:16]


class TestAGT006:

    def test_006_email_fires(self):
        """An email address in arguments must fire AGT-006."""
        call = make_call("send_message", {"body": "Contact user@example.com for info"})
        result = audit(call)
        assert "AGT-006" in ids_in(result)

    def test_006_ssn_fires(self):
        """An SSN pattern in arguments must fire AGT-006."""
        call = make_call("submit_form", {"data": "SSN: 123-45-6789"})
        result = audit(call)
        assert "AGT-006" in ids_in(result)

    def test_006_credit_card_fires(self):
        """A credit card number in arguments must fire AGT-006."""
        call = make_call("process_payment", {"card": "4111 1111 1111 1111"})
        result = audit(call)
        assert "AGT-006" in ids_in(result)

    def test_006_credit_card_dashes_fires(self):
        """A credit card with dashes in arguments must fire AGT-006."""
        call = make_call("process_payment", {"card": "4111-1111-1111-1111"})
        result = audit(call)
        assert "AGT-006" in ids_in(result)

    def test_006_credit_card_no_separator_fires(self):
        """A credit card with no separator in arguments must fire AGT-006."""
        call = make_call("process_payment", {"card": "4111111111111111"})
        result = audit(call)
        assert "AGT-006" in ids_in(result)

    def test_006_api_key_fires(self):
        """An AWS-style API key in arguments must fire AGT-006."""
        call = make_call("configure_aws", {"key": _FAKE_API_KEY})
        result = audit(call)
        assert "AGT-006" in ids_in(result)

    def test_006_api_key_in_longer_string_fires(self):
        """An API key embedded in a longer string must fire AGT-006."""
        call = make_call("configure_aws", {"config": f"key={_FAKE_API_KEY}&region=us-east-1"})
        result = audit(call)
        assert "AGT-006" in ids_in(result)

    def test_006_clean_args_no_fire(self):
        """Normal safe arguments must not fire AGT-006."""
        call = make_call("search_tool", {"query": "latest security news"})
        result = audit(call)
        assert "AGT-006" not in ids_in(result)

    def test_006_severity_is_high(self):
        """AGT-006 finding must carry HIGH severity."""
        call = make_call("send_message", {"body": "email: admin@corp.com"})
        result = audit(call)
        finding = next(f for f in result.findings if f.check_id == "AGT-006")
        assert finding.severity == "HIGH"

    def test_006_weight_is_25(self):
        """AGT-006 weight must be 25."""
        assert _CHECK_WEIGHTS["AGT-006"] == 25

    def test_006_fires_once_per_call(self):
        """Even with multiple PII patterns, AGT-006 fires only once per call."""
        call = make_call("log_tool", {
            "a": "user@example.com",
            "b": "123-45-6789",
        })
        result = audit(call)
        agt006_findings = [f for f in result.findings if f.check_id == "AGT-006"]
        assert len(agt006_findings) == 1

    def test_006_empty_args_no_fire(self):
        """Empty argument values must not fire AGT-006."""
        call = make_call("tool", {"key": ""})
        result = audit(call)
        assert "AGT-006" not in ids_in(result)

    def test_006_no_args_no_fire(self):
        """No arguments must not fire AGT-006."""
        call = make_call("tool", {})
        result = audit(call)
        assert "AGT-006" not in ids_in(result)


# ===========================================================================
# AGT-007: Excessive call chain depth
# ===========================================================================

class TestAGT007:

    def test_007_depth_5_no_fire(self):
        """Depth exactly 5 must not fire AGT-007."""
        call = make_call("any_tool", depth=5)
        result = audit(call)
        assert "AGT-007" not in ids_in(result)

    def test_007_depth_6_fires(self):
        """Depth 6 must fire AGT-007."""
        call = make_call("any_tool", depth=6)
        result = audit(call)
        assert "AGT-007" in ids_in(result)

    def test_007_depth_0_no_fire(self):
        """Depth 0 must not fire AGT-007."""
        call = make_call("any_tool", depth=0)
        result = audit(call)
        assert "AGT-007" not in ids_in(result)

    def test_007_depth_100_fires(self):
        """A very large depth must fire AGT-007."""
        call = make_call("any_tool", depth=100)
        result = audit(call)
        assert "AGT-007" in ids_in(result)

    def test_007_depth_4_no_fire(self):
        """Depth 4 must not fire AGT-007."""
        call = make_call("any_tool", depth=4)
        result = audit(call)
        assert "AGT-007" not in ids_in(result)

    def test_007_depth_7_fires(self):
        """Depth 7 must fire AGT-007."""
        call = make_call("any_tool", depth=7)
        result = audit(call)
        assert "AGT-007" in ids_in(result)

    def test_007_severity_is_medium(self):
        """AGT-007 finding must carry MEDIUM severity."""
        call = make_call("any_tool", depth=6)
        result = audit(call)
        finding = next(f for f in result.findings if f.check_id == "AGT-007")
        assert finding.severity == "MEDIUM"

    def test_007_weight_is_15(self):
        """AGT-007 weight must be 15."""
        assert _CHECK_WEIGHTS["AGT-007"] == 15

    def test_007_single_finding_per_call(self):
        """AGT-007 must produce exactly one finding per deep call."""
        call = make_call("any_tool", depth=10)
        result = audit(call)
        agt007_findings = [f for f in result.findings if f.check_id == "AGT-007"]
        assert len(agt007_findings) == 1


# ===========================================================================
# AGTResult model — risk_score, action, helpers
# ===========================================================================

class TestAGTResult:

    def test_result_no_findings_allow(self):
        """A clean call must yield action=ALLOW and risk_score=0."""
        call = make_call("safe_tool", {"key": "value"})
        result = audit(call)
        assert result.action == "ALLOW"
        assert result.risk_score == 0
        assert result.findings == []

    def test_result_action_warn_threshold(self):
        """Risk score >= 30 but < 70 must yield action=WARN."""
        # AGT-007 alone fires weight 15; need 30+ for WARN
        # AGT-002 (25) + AGT-007 (15) = 40 => WARN
        history = [make_call("t", call_id=f"h{i}", timestamp_ms=i) for i in range(10)]
        call = make_call("any_tool", timestamp_ms=5000, depth=6)
        result = audit(call, call_history=history, burst_limit=10, burst_window_ms=100000)
        assert result.action == "WARN"
        assert 30 <= result.risk_score < 70

    def test_result_action_block_threshold(self):
        """Risk score >= 70 must yield action=BLOCK."""
        # AGT-001 (45) + AGT-005 (40) = 85 => BLOCK
        # Tool name contains "bash" (exec keyword) AND "write_file" (FS keyword)
        call = make_call("bash_write_file", {"cmd": "ls /", "path": "/etc/shadow"})
        result = audit(call)
        assert result.action == "BLOCK"
        assert result.risk_score >= 70

    def test_result_risk_score_capped_at_100(self):
        """Combined weights exceeding 100 must be capped at 100."""
        # AGT-001 (45) + AGT-005 (40) + AGT-006 (25) = 110 => capped at 100
        # Tool name contains "bash" (exec keyword) AND "write_file" (FS keyword)
        call = make_call(
            "bash_write_file",
            {"cmd": "cat /etc/shadow", "path": "/etc/passwd", "user": "admin@corp.com"},
        )
        result = audit(call)
        assert result.risk_score <= 100

    def test_result_to_dict_keys(self):
        """to_dict() must contain risk_score, action, and findings keys."""
        call = make_call("safe_tool")
        result = audit(call)
        d = result.to_dict()
        assert "risk_score" in d
        assert "action" in d
        assert "findings" in d

    def test_result_to_dict_findings_structure(self):
        """Each finding dict in to_dict() must have the expected keys."""
        call = make_call("any_tool", depth=6)
        result = audit(call)
        for item in result.to_dict()["findings"]:
            assert "check_id" in item
            assert "severity" in item
            assert "title" in item
            assert "detail" in item
            assert "weight" in item
            assert "call_id" in item

    def test_result_summary_contains_action(self):
        """summary() must include the action string."""
        call = make_call("safe_tool")
        result = audit(call)
        assert "ALLOW" in result.summary()

    def test_result_summary_contains_risk_score(self):
        """summary() must include the risk score."""
        call = make_call("any_tool", depth=6)
        result = audit(call)
        assert str(result.risk_score) in result.summary()

    def test_result_by_severity_grouping(self):
        """by_severity() must correctly group findings by severity level."""
        call = make_call("execute_bash", {"cmd": "ls /", "path": "/etc/shadow"})
        result = audit(call)
        grouped = result.by_severity()
        for severity, findings in grouped.items():
            for f in findings:
                assert f.severity == severity

    def test_result_finding_call_id_matches(self):
        """Each finding's call_id must match the audited call's call_id."""
        call = make_call("any_tool", call_id="my-unique-id", depth=6)
        result = audit(call)
        for f in result.findings:
            assert f.call_id == "my-unique-id"

    def test_result_unique_checks_counted_once(self):
        """A check ID that fires must contribute its weight only once to risk_score."""
        # AGT-007 weight=15, depth=6 fires once
        call = make_call("any_tool", depth=6)
        result = audit(call)
        assert result.risk_score == 15

    def test_result_action_allow_below_30(self):
        """Risk score below 30 must yield ALLOW."""
        # Only AGT-007 fires: score=15 => ALLOW
        call = make_call("any_tool", depth=6)
        result = audit(call)
        assert result.action == "ALLOW"
        assert result.risk_score == 15


# ===========================================================================
# audit_sequence
# ===========================================================================

class TestAuditSequence:

    def test_sequence_empty_returns_empty(self):
        """An empty call list must return an empty result list."""
        results = audit_sequence([])
        assert results == []

    def test_sequence_length_matches_input(self):
        """audit_sequence must return one result per input call."""
        calls = [make_call("tool", call_id=f"c{i}", timestamp_ms=i * 1000) for i in range(5)]
        results = audit_sequence(calls)
        assert len(results) == 5

    def test_sequence_first_call_has_no_history(self):
        """The first call in a sequence has no history; burst cannot exceed 1."""
        calls = [make_call("any_tool", call_id="c0", timestamp_ms=0)]
        results = audit_sequence(calls, burst_limit=1)
        # count=1, burst_limit=1 => not over
        assert "AGT-002" not in ids_in(results[0])

    def test_sequence_burst_detected_mid_sequence(self):
        """Burst AGT-002 must fire when the cumulative count exceeds burst_limit."""
        # burst_limit=3; 4th call (index 3) should see count=4 > 3 => fires
        calls = [
            make_call("any_tool", call_id=f"c{i}", timestamp_ms=i * 100)
            for i in range(5)
        ]
        results = audit_sequence(calls, burst_limit=3, burst_window_ms=10000)
        assert "AGT-002" not in ids_in(results[0])
        assert "AGT-002" not in ids_in(results[1])
        assert "AGT-002" not in ids_in(results[2])
        assert "AGT-002" in ids_in(results[3])

    def test_sequence_loop_detected_after_enough_repeats(self):
        """AGT-003 must only fire once there are 3 prior identical calls."""
        same_args = {"query": "repeat"}
        calls = [
            make_call("search_tool", same_args, call_id=f"c{i}", timestamp_ms=i * 100)
            for i in range(5)
        ]
        results = audit_sequence(calls)
        # Index 0,1,2: not enough history to fire
        assert "AGT-003" not in ids_in(results[0])
        assert "AGT-003" not in ids_in(results[1])
        assert "AGT-003" not in ids_in(results[2])
        # Index 3: has 3 identical history calls => fires
        assert "AGT-003" in ids_in(results[3])

    def test_sequence_each_result_is_agtresult(self):
        """Each element in audit_sequence output must be an AGTResult instance."""
        calls = [make_call("tool", call_id=f"c{i}") for i in range(3)]
        results = audit_sequence(calls)
        for r in results:
            assert isinstance(r, AGTResult)

    def test_sequence_passes_allowed_dirs(self):
        """allowed_working_dirs must be forwarded to each call audit."""
        calls = [
            make_call("read_file", {"path": "/etc/passwd"}, call_id="c0"),
            make_call("read_file", {"path": "/workspace/main.py"}, call_id="c1"),
        ]
        results = audit_sequence(calls, allowed_working_dirs=["/workspace"])
        assert "AGT-001" in ids_in(results[0])
        assert "AGT-001" not in ids_in(results[1])

    def test_sequence_passes_allowed_domains(self):
        """allowed_domains must be forwarded to each call audit."""
        calls = [
            make_call("http_get", {"url": "https://evil.com/data"}, call_id="c0"),
            make_call("http_get", {"url": "https://safe.com/data"}, call_id="c1"),
        ]
        results = audit_sequence(calls, allowed_domains=["safe.com"])
        assert "AGT-004" in ids_in(results[0])
        assert "AGT-004" not in ids_in(results[1])

    def test_sequence_independent_results(self):
        """Each audit result in a sequence must be independent (different call_ids)."""
        calls = [make_call("any_tool", depth=6, call_id=f"c{i}") for i in range(3)]
        results = audit_sequence(calls)
        for i, r in enumerate(results):
            for f in r.findings:
                assert f.call_id == f"c{i}"


# ===========================================================================
# Combined / multi-check scenarios
# ===========================================================================

class TestCombinedChecks:

    def test_combined_agt001_and_agt005(self):
        """A call that triggers both AGT-001 and AGT-005 must report both."""
        # Tool name contains "bash" (exec keyword) AND "write_file" (FS keyword)
        call = make_call(
            "bash_write_file",
            {"cmd": "rm -rf /", "path": "/etc/shadow"},
        )
        result = audit(call)
        assert "AGT-001" in ids_in(result)
        assert "AGT-005" in ids_in(result)

    def test_combined_risk_score_sums_correctly(self):
        """Risk score must equal the sum of weights of unique fired checks."""
        # Only AGT-007 (15) fires
        call = make_call("any_tool", depth=7)
        result = audit(call)
        assert result.risk_score == 15

    def test_combined_all_checks_clean_call(self):
        """A completely clean call must produce zero findings and ALLOW action."""
        call = make_call("list_documents", {"folder": "reports"}, depth=1)
        result = audit(call)
        assert len(result.findings) == 0
        assert result.action == "ALLOW"
        assert result.risk_score == 0

    def test_combined_agt006_and_agt007(self):
        """PII in args AND excessive depth must both fire."""
        call = make_call("log_event", {"data": "user@test.com"}, depth=6)
        result = audit(call)
        assert "AGT-006" in ids_in(result)
        assert "AGT-007" in ids_in(result)
        # 25 + 15 = 40 => WARN
        assert result.action == "WARN"

    def test_combined_agt005_block(self):
        """AGT-005 (weight=40) + AGT-001 (weight=45) = 85 => BLOCK."""
        # Tool name contains "bash" (exec keyword) AND "read_file" (FS keyword)
        call = make_call("bash_read_file", {"cmd": "ls", "path": "/proc/self"})
        result = audit(call)
        assert result.action == "BLOCK"
