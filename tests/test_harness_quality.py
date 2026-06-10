"""Tests for the harness_quality scoring engine.

Covers: clean session, heavy-error session, disaster cap, monotonicity,
size normalization, saturation fixed, error detection, duplicate normalization,
first_prompt title cleaning, and grade boundaries.
"""

import json
from pathlib import Path

from claudesavvy.analyzers.harness_quality import (
    CONFIG,
    _clean_user_text,
    _normalize_cmd,
    build_suggestions,
    evaluate_file,
    first_prompt,
    format_tuning_report,
    score_session,
)


# ---------------------------------------------------------------------------
# Helpers to build parsed-transcript dicts consumed by score_session()
# ---------------------------------------------------------------------------

def _make_parsed(
    tool_calls=None,   # list of (name, command_or_path, is_error, rejected)
    assistant_turns=5,
):
    """
    Build the dict shape that score_session() expects.

    tool_calls: list of tuples
        (tool_name, command_or_path, is_error: bool, rejected: bool)
    """
    tool_uses = {}
    results = []
    for i, (name, cmd, is_error, rejected) in enumerate(tool_calls or []):
        tid = f"t{i}"
        inp = {}
        if name == "Bash":
            inp = {"command": cmd}
        elif name == "Read":
            inp = {"file_path": cmd}
        tool_uses[tid] = {"name": name, "command": cmd, "input": inp}
        results.append({"id": tid, "is_error": is_error, "rejected": rejected, "text": ""})
    return {
        "tool_uses": tool_uses,
        "results": results,
        "assistant_turns": assistant_turns,
        "subagent_count": 0,
    }


def _make_parsed_with_text(tool_calls_with_text):
    """
    Like _make_parsed but takes (name, cmd, text) where text is the result text.
    is_error is determined by error_text_re (no is_error key means fallback to text).
    """
    tool_uses = {}
    results = []
    for i, (name, cmd, text) in enumerate(tool_calls_with_text):
        tid = f"t{i}"
        inp = {"command": cmd} if name == "Bash" else {"file_path": cmd}
        tool_uses[tid] = {"name": name, "command": cmd, "input": inp}
        # Don't set is_error: let the regex decide
        results.append({"id": tid, "rejected": False, "text": text})
    return {
        "tool_uses": tool_uses,
        "results": results,
        "assistant_turns": 3,
        "subagent_count": 0,
    }


# ---------------------------------------------------------------------------
# 1. Clean session → score 100, grade A, no-issue suggestion
# ---------------------------------------------------------------------------

def test_clean_session():
    parsed = _make_parsed([
        ("Bash", "ls", False, False),
        ("Read", "/foo/bar.py", False, False),
        ("Bash", "git status", False, False),
        ("Bash", "echo hi", False, False),
    ])
    result = score_session(parsed)
    assert result["score"] == 100
    assert result["grade"] == "A"
    assert result["low_activity"] is False

    suggestions = build_suggestions(result)
    assert len(suggestions) == 1
    assert "Clean run" in suggestions[0]


# ---------------------------------------------------------------------------
# 2. Heavy-error session → low score but > 0; disaster cap behavior
# ---------------------------------------------------------------------------

def test_heavy_errors_score_above_zero():
    """57 errors out of 320 calls should score below 100 but NOT at 0."""
    # 57 errors + 263 clean calls = 320 total tool calls (17.8% error rate)
    # With errors weight=0.30 and ref=0.50, error subscore ~64.
    # Other categories are clean (no dups, no repeat reads, etc.).
    # Overall = 0.30 * 64 + 0.70 * 100 ~= 89 → below 100 but not catastrophic alone.
    calls = [("Bash", f"cmd{i}", True, False) for i in range(57)]
    calls += [("Bash", f"clean{i}", False, False) for i in range(263)]
    parsed = _make_parsed(calls)
    result = score_session(parsed)
    assert result["score"] > 0, "Score should not pin at 0 for non-catastrophic error rate"
    assert result["score"] < 100, "57/320 errors should produce a score below 100"
    # Errors subscore should be significantly below 100
    assert result["subscores"]["errors"] < 70, (
        f"Error subscore should reflect heavy errors, got {result['subscores']['errors']}"
    )


def test_disaster_cap_applied():
    """
    If errors subscore is 0 (all calls errored), overall should be <= 0 + 45 = 45.
    Other categories are perfect (no dups, no repeat reads, 1 bash call).
    """
    # 100% error rate → errors subscore = 0 → disaster cap = 0 + 45 = 45
    calls = [("Bash", f"fail{i}", True, False) for i in range(10)]
    parsed = _make_parsed(calls)
    result = score_session(parsed)
    assert result["subscores"]["errors"] == 0.0
    assert result["score"] <= 45, f"Disaster cap should hold score ≤ 45, got {result['score']}"


def test_disaster_cap_only_guards_errors_and_rejections():
    """
    A session whose only flaw is scriptability (one binary hammered, zero
    errors/dups/rejections) must not be capped to an F — efficiency
    categories cost at most their weight.
    """
    calls = [("Bash", f"git log --oneline f{i}", False, False) for i in range(30)]
    parsed = _make_parsed(calls)
    result = score_session(parsed)
    assert result["subscores"]["scriptability"] == 0.0
    # 15% weight → overall 85, not dragged to ≤45 by the disaster guard
    assert result["score"] == 85
    assert result["grade"] == "B"


def test_disaster_cap_not_applied_on_clean():
    """Clean session: disaster cap does not reduce a perfect score."""
    calls = [("Bash", f"cmd{i}", False, False) for i in range(10)]
    parsed = _make_parsed(calls)
    result = score_session(parsed)
    assert result["score"] == 100


# ---------------------------------------------------------------------------
# 3. Monotonicity: more errors ⇒ score ≤ fewer errors
# ---------------------------------------------------------------------------

def test_monotonicity_errors():
    def make(n_errors, total=50):
        calls = [("Bash", f"cmd{i}", i < n_errors, False) for i in range(total)]
        return score_session(_make_parsed(calls))["score"]

    s0 = make(0)
    s5 = make(5)
    s20 = make(20)
    assert s0 >= s5, "No errors should score at least as well as 5 errors"
    assert s5 >= s20, "5 errors should score at least as well as 20 errors"


# ---------------------------------------------------------------------------
# 4. Size normalization: same error count, more tool calls ⇒ higher score
# ---------------------------------------------------------------------------

def test_size_normalization():
    def make(n_errors, total):
        calls = [("Bash", f"cmd{i}", i < n_errors, False) for i in range(total)]
        return score_session(_make_parsed(calls))["score"]

    # 10 errors in 15 calls vs 10 errors in 100 calls
    score_small = make(10, 15)
    score_large = make(10, 100)
    assert score_large > score_small, (
        f"10 errors in 100 calls ({score_large}) should score better than "
        f"10 errors in 15 calls ({score_small})"
    )


# ---------------------------------------------------------------------------
# 5. Saturation fixed: two terrible sessions get different scores
# ---------------------------------------------------------------------------

def test_saturation_different_scores():
    """
    Two sessions that under the old model would both pin at 0 should get
    different scores under the new model.
    """
    # Session A: 80% error rate
    calls_a = [("Bash", f"cmd{i}", i < 80, False) for i in range(100)]
    # Session B: 30% error rate (still bad, but less catastrophic)
    calls_b = [("Bash", f"cmd{i}", i < 30, False) for i in range(100)]

    score_a = score_session(_make_parsed(calls_a))["score"]
    score_b = score_session(_make_parsed(calls_b))["score"]

    assert score_b > score_a, (
        f"30% error rate ({score_b}) should score better than 80% ({score_a})"
    )
    # Both should be distinguishable (not both 0)
    assert not (score_a == 0 and score_b == 0), "Two different bad sessions pinned both at 0"


# ---------------------------------------------------------------------------
# 6. Error detection: is_error flag and strict text patterns
# ---------------------------------------------------------------------------

def test_is_error_flag_counts():
    """is_error=True should always count as an error."""
    calls = [("Bash", "whatever", True, False) for _ in range(5)]
    calls += [("Bash", "ok", False, False) for _ in range(5)]
    parsed = _make_parsed(calls)
    result = score_session(parsed)
    assert result["metrics"]["tool_errors"] == 5


def test_benign_text_no_error():
    """'No errors found' text (no is_error key) should NOT count as an error."""
    parsed = _make_parsed_with_text([
        ("Bash", "check", "No errors found"),
        ("Bash", "lint", "0 errors, 0 warnings"),
        ("Bash", "test", "All tests passed"),
    ])
    result = score_session(parsed)
    assert result["metrics"]["tool_errors"] == 0, (
        "Benign 'No errors found' text should not trigger error detection"
    )


def test_traceback_text_counts_as_error():
    """'Traceback (most recent call last):' should count as error via text fallback."""
    parsed = _make_parsed_with_text([
        ("Bash", "python script.py", "Traceback (most recent call last):\n  File foo.py\nValueError: oops"),
        ("Bash", "clean", "done"),
        ("Bash", "also_clean", "ok"),
        ("Bash", "more_clean", "fine"),
    ])
    result = score_session(parsed)
    assert result["metrics"]["tool_errors"] == 1, (
        "Traceback text should be detected as an error"
    )


def test_error_colon_at_line_start_counts():
    """Line-start 'Error: ...' should count as error via text fallback."""
    parsed = _make_parsed_with_text([
        ("Bash", "run", "Error: something failed"),
        ("Bash", "ok", "success"),
        ("Bash", "ok2", "fine"),
        ("Bash", "ok3", "great"),
    ])
    result = score_session(parsed)
    assert result["metrics"]["tool_errors"] == 1


def test_command_not_found_counts():
    parsed = _make_parsed_with_text([
        ("Bash", "foobar", "bash: foobar: command not found"),
        ("Bash", "ok", "done"),
        ("Bash", "ok2", "done"),
        ("Bash", "ok3", "done"),
    ])
    result = score_session(parsed)
    assert result["metrics"]["tool_errors"] == 1


# ---------------------------------------------------------------------------
# 7. Duplicate normalization: 'git  status' vs 'git status' are duplicates
# ---------------------------------------------------------------------------

def test_duplicate_normalization():
    """Commands that differ only in whitespace should be treated as duplicates."""
    parsed = _make_parsed([
        ("Bash", "git status", False, False),
        ("Bash", "git  status", False, False),   # extra space
        ("Bash", "git   status", False, False),  # triple space
        ("Bash", "ls", False, False),
    ])
    result = score_session(parsed)
    dups = result["detail"]["duplicates"]
    assert "git status" in dups, "Whitespace-normalized duplicates should be detected"
    assert dups["git status"] == 3, f"Expected 3 occurrences, got {dups}"


# ---------------------------------------------------------------------------
# 8. first_prompt title cleaning
# ---------------------------------------------------------------------------

def _write_jsonl(tmp_path: Path, entries: list) -> Path:
    """Write a list of dicts as JSONL to a temp file, return the path."""
    p = tmp_path / "session.jsonl"
    with p.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return p


def test_first_prompt_skips_caveat(tmp_path):
    """Messages starting with 'Caveat: The messages below...' should be skipped."""
    path = _write_jsonl(tmp_path, [
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "text",
                     "text": "Caveat: The messages below were generated by the user while running local commands in their IDE.\n<command-name>ls</command-name>"}
                ]
            }
        },
        {
            "type": "user",
            "message": {
                "content": [{"type": "text", "text": "Please refactor the auth module."}]
            }
        }
    ])
    result = first_prompt(path)
    assert result == "Please refactor the auth module.", f"Got: {result!r}"


def test_first_prompt_skips_system_reminder_only(tmp_path):
    """Messages consisting only of system-reminder tags should be skipped."""
    path = _write_jsonl(tmp_path, [
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "text",
                     "text": "<system-reminder>Do something</system-reminder>"}
                ]
            }
        },
        {
            "type": "user",
            "message": {
                "content": [{"type": "text", "text": "Fix the login bug."}]
            }
        }
    ])
    result = first_prompt(path)
    assert result == "Fix the login bug.", f"Got: {result!r}"


def test_first_prompt_strips_caveat_prefix_and_keeps_rest(tmp_path):
    """Mixed caveat + real content: strip caveat paragraph, return the rest."""
    path = _write_jsonl(tmp_path, [
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "text",
                     "text": "Caveat: The messages below were generated by the user while running local commands in their IDE.\nPlease add dark mode support."}
                ]
            }
        }
    ])
    result = first_prompt(path)
    assert "Please add dark mode support" in result, f"Got: {result!r}"
    assert "Caveat" not in result, "Caveat prefix should be stripped"


def test_first_prompt_returns_real_prompt(tmp_path):
    """Normal user message should be returned as-is."""
    path = _write_jsonl(tmp_path, [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello"}]}},
        {
            "type": "user",
            "message": {
                "content": [{"type": "text", "text": "Implement pagination for the API."}]
            }
        }
    ])
    result = first_prompt(path)
    assert result == "Implement pagination for the API.", f"Got: {result!r}"


# ---------------------------------------------------------------------------
# 9. _clean_user_text directly
# ---------------------------------------------------------------------------

def test_clean_user_text_removes_command_name():
    raw = "<command-name>git status</command-name> something else"
    result = _clean_user_text(raw)
    assert "command-name" not in result
    assert "git status" not in result
    assert "something else" in result


def test_clean_user_text_strips_caveat():
    raw = "Caveat: The messages below were generated by the user while running local commands in their IDE.\nReal content here."
    result = _clean_user_text(raw)
    assert "Caveat" not in result
    assert "Real content here" in result


def test_clean_user_text_empty_after_strip():
    """Pure infrastructure message → empty after clean."""
    raw = "<system-reminder>remember something</system-reminder>"
    result = _clean_user_text(raw)
    assert result == ""


# ---------------------------------------------------------------------------
# 10. Grade boundaries
# ---------------------------------------------------------------------------

def test_grade_boundaries():
    """Score boundaries map to correct letter grades."""
    grade_map = CONFIG["grades"]

    def grade_for(score):
        return next(g for floor, g in grade_map if score >= floor)

    assert grade_for(100) == "A"
    assert grade_for(90) == "A"
    assert grade_for(89) == "B"
    assert grade_for(80) == "B"
    assert grade_for(79) == "C"
    assert grade_for(70) == "C"
    assert grade_for(69) == "D"
    assert grade_for(60) == "D"
    assert grade_for(59) == "F"
    assert grade_for(0) == "F"


# ---------------------------------------------------------------------------
# 11. Low activity flag
# ---------------------------------------------------------------------------

def test_low_activity_no_issues():
    """Under 3 tool calls with no errors → score 100, low_activity True."""
    parsed = _make_parsed([("Bash", "ls", False, False)])
    result = score_session(parsed)
    assert result["low_activity"] is True
    assert result["score"] == 100


def test_low_activity_with_errors():
    """Under 3 tool calls with errors → score 60, low_activity True."""
    parsed = _make_parsed([("Bash", "fail", True, False)])
    result = score_session(parsed)
    assert result["low_activity"] is True
    assert result["score"] == 60


# ---------------------------------------------------------------------------
# 12. Normalize command helper
# ---------------------------------------------------------------------------

def test_normalize_cmd():
    assert _normalize_cmd("git  status") == "git status"
    assert _normalize_cmd("  ls   -la  ") == "ls -la"
    assert _normalize_cmd("echo  hello   world") == "echo hello world"


# ---------------------------------------------------------------------------
# 13. Example session from spec: 57 errors / 320 calls → D-F range, not 0
# ---------------------------------------------------------------------------

def test_spec_example_session():
    """
    The spec's example: 57 errors / 320 tool calls, 12 extra dups / 131 bash,
    18 repeat reads, a few clusters, 0 rejections → score should be > 0
    and clearly below a clean session.

    The combined effect of multiple penalized categories should produce a score
    substantially below 100. With ref_rates as configured, each category
    contributes penalty, pulling the overall score down significantly.
    """
    calls = []
    # 57 error calls (bash)
    for i in range(57):
        calls.append(("Bash", f"bad_cmd_{i}", True, False))
    # 12 duplicate pairs (each run twice — 12 extra)
    for i in range(12):
        calls.append(("Bash", f"dup_cmd_{i}", False, False))
        calls.append(("Bash", f"dup_cmd_{i}", False, False))
    # Fill up to 320 total with unique clean bash calls
    total_so_far = len(calls)
    for i in range(320 - total_so_far):
        calls.append(("Bash", f"unique_{i}", False, False))

    parsed = _make_parsed(calls)
    result = score_session(parsed)

    assert result["score"] > 0, f"Score should not be 0, got {result['score']}"
    assert result["score"] < 100, "Session with 57 errors and 12 dup extra should not score 100"
    # Should be noticeably penalized across categories
    assert result["subscores"]["errors"] < 100, "Error subscore should be < 100"
    assert result["subscores"]["duplicate_commands"] < 100, "Dup subscore should be < 100"


# ---------------------------------------------------------------------------
# 14. evaluate_file: integration smoke test via tmp_path
# ---------------------------------------------------------------------------

def test_evaluate_file_smoke(tmp_path):
    """evaluate_file should return a dict with the expected keys."""
    path = _write_jsonl(tmp_path, [
        {"type": "user", "message": {"content": [{"type": "text", "text": "Do the thing."}]}},
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Bash",
                     "input": {"command": "ls"}}
                ]
            }
        },
        {
            "type": "tool",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1",
                     "is_error": False, "content": "file.txt"}
                ]
            }
        },
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "id": "t2", "name": "Bash",
                     "input": {"command": "git status"}}
                ]
            }
        },
        {
            "type": "tool",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "t2",
                     "is_error": False, "content": "nothing to commit"}
                ]
            }
        },
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "id": "t3", "name": "Bash",
                     "input": {"command": "echo done"}}
                ]
            }
        },
        {
            "type": "tool",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "t3",
                     "is_error": False, "content": "done"}
                ]
            }
        },
    ])
    record = evaluate_file(path)
    assert record is not None
    assert "score" in record
    assert "grade" in record
    assert "subscores" in record
    assert "weights" in record
    assert "rates" in record
    assert "metrics" in record
    assert "suggestions" in record
    assert "detail" in record
    assert "low_activity" in record
    assert "penalties" not in record, "Old 'penalties' key should not be present"


# ---------------------------------------------------------------------------
# 12. Tuning report for the copy-into-Claude-Code button
# ---------------------------------------------------------------------------

def test_format_tuning_report_contents():
    calls = (
        [("Bash", "make test", True, False)] * 3
        + [("Bash", "git status", False, False)] * 2
        + [("Read", "/proj/config.py", False, False)] * 3
        + [("Grep", "", False, False)] * 4
    )
    parsed = _make_parsed(calls)
    scored = score_session(parsed)
    record = {
        "session_id": "abc123",
        "project_label": "my-project",
        "date": "2026-06-10 09:00",
        "prompt": "fix the build",
        "score": scored["score"],
        "grade": scored["grade"],
        "low_activity": scored["low_activity"],
        "metrics": scored["metrics"],
        "subscores": scored["subscores"],
        "weights": scored["weights"],
        "rates": scored["rates"],
        "suggestions": build_suggestions(scored),
        "detail": scored["detail"],
    }
    report = format_tuning_report(record)
    # Instructional preamble so it's directly actionable in Claude Code
    assert "harness improvements" in report
    assert "CLAUDE.md" in report
    # Session identity and overall result
    assert "abc123" in report
    assert "my-project" in report
    assert "fix the build" in report
    assert f"{scored['score']}/100 (grade {scored['grade']})" in report
    # Category breakdown with rates
    assert "Tool errors" in report
    assert "3/12 errors / tool calls" in report
    # Offenders: the duplicated command and the re-read file appear
    assert "make test" in report
    assert "/proj/config.py" in report
    # Impact-ranked suggestions present
    assert "## Tuning suggestions (impact-ranked)" in report
