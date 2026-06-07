"""Harness quality analyzer — scores Claude Code session transcripts.

Walks a session JSONL transcript, attributes tool errors, detects wasteful
patterns (duplicate commands, repeated reads, low scriptability, user
rejections), and produces a 0-100 score with a letter grade and
skill-tuning suggestions.

Ported and extended from a standalone session-scoring script. No third-party
deps; defensive against transcript schema drift.
"""

import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

# ----------------------------------------------------------------------------
# Tunable scoring config. Transparent — edit weights to taste.
# ----------------------------------------------------------------------------
CONFIG = {
    "start_score": 100,
    "weights": {
        "error_per_hit": 6,          # each errored tool result
        "error_cap": 40,             # max total deduction from errors
        "dup_command_per_extra": 4,  # each repeat of an identical command beyond the first
        "dup_command_cap": 30,
        "repeat_read_per_extra": 2,  # each re-read of the same file beyond the first
        "repeat_read_cap": 15,
        "scriptability_per_cluster": 5,  # each binary called too many times raw
        "scriptability_cap": 20,
        "rejection_per_hit": 8,      # each tool call the user rejected
        "rejection_cap": 32,
    },
    # if a single shell binary is invoked this many times, it should probably be
    # a script / a single batched call instead.
    "scriptability_threshold": 5,
    # error text fallback, only used when a tool_result has no explicit is_error key
    "error_text_re": re.compile(
        r"(?i)\b(error|traceback|exception|command not found|no such file|"
        r"fatal:|permission denied|segmentation fault)\b"
    ),
    # Claude Code marks a denied tool call with this canned result text.
    "rejection_text_re": re.compile(
        r"(?i)(the user doesn'?t want to (?:proceed|take this action)|"
        r"user rejected|tool use was rejected|operation was (?:rejected|cancelled) by the user)"
    ),
    "grades": [  # (min_score, letter)
        (90, "A"), (80, "B"), (70, "C"), (60, "D"), (0, "F"),
    ],
}

TOOL_BLOCK_TYPES = {"tool_use", "tool_result"}


# ----------------------------------------------------------------------------
# Transcript parsing — defensive against schema drift.
# ----------------------------------------------------------------------------
def _iter_lines(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _walk_blocks(obj):
    """Yield every dict that looks like a tool_use / tool_result block, anywhere."""
    if isinstance(obj, dict):
        if obj.get("type") in TOOL_BLOCK_TYPES:
            yield obj
        for v in obj.values():
            yield from _walk_blocks(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_blocks(v)


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                if "text" in b:
                    parts.append(str(b.get("text", "")))
                elif b.get("type") == "tool_result":
                    parts.append(_text_of(b.get("content")))
        return "\n".join(parts)
    return ""


def _first_token(cmd: str) -> str:
    """Best-effort 'which binary' for a shell command, ignoring env prefixes."""
    toks = cmd.strip().split()
    i = 0
    while i < len(toks) and ("=" in toks[i] and not toks[i].startswith("-")):
        i += 1  # skip VAR=val prefixes
    return toks[i] if i < len(toks) else ""


def _parse_one(path: Path, tool_uses: dict, results: list) -> int:
    """Parse a transcript file into shared tool_uses/results. Returns assistant_turns."""
    assistant_turns = 0
    for entry in _iter_lines(path):
        if entry.get("type") == "assistant":
            assistant_turns += 1
        msg = entry.get("message", entry)
        for block in _walk_blocks(msg):
            if block.get("type") == "tool_use":
                name = block.get("name", "?")
                inp = block.get("input") or {}
                cmd = ""
                if isinstance(inp, dict):
                    cmd = inp.get("command") or inp.get("file_path") or ""
                tool_uses[block.get("id")] = {
                    "name": name,
                    "command": cmd,
                    "input": inp if isinstance(inp, dict) else {},
                }
            elif block.get("type") == "tool_result":
                rid = block.get("tool_use_id")
                txt = _text_of(block.get("content"))
                if "is_error" in block:
                    err = bool(block["is_error"])
                else:
                    err = bool(CONFIG["error_text_re"].search(txt))
                rejected = bool(CONFIG["rejection_text_re"].search(txt))
                results.append({"id": rid, "is_error": err, "rejected": rejected, "text": txt})
    return assistant_turns


def parse_transcript(path: Path) -> dict:
    """Parse a session transcript (and its subagents) into tool/result records."""
    tool_uses: dict = {}
    results: list = []

    assistant_turns = _parse_one(path, tool_uses, results)

    # Merge subagent transcripts at <project>/<session-id>/subagents/agent-*.jsonl
    subagents_dir = path.parent / path.stem / "subagents"
    subagent_files = sorted(subagents_dir.glob("agent-*.jsonl")) if subagents_dir.is_dir() else []
    for sub in subagent_files:
        assistant_turns += _parse_one(sub, tool_uses, results)

    return {
        "tool_uses": tool_uses,
        "results": results,
        "assistant_turns": assistant_turns,
        "subagent_count": len(subagent_files),
    }


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def score_session(parsed: dict) -> dict:
    """Score a parsed transcript into a 0-100 result with penalties and detail."""
    tool_uses = parsed["tool_uses"]
    results = parsed["results"]
    w = CONFIG["weights"]

    # --- errors, attributed to the command that produced them ---
    errors = []
    rejections = []
    for r in results:
        tu = tool_uses.get(r["id"], {})
        label = tu.get("command") or tu.get("name") or "?"
        snippet = r["text"].strip().splitlines()[0][:160] if r["text"].strip() else ""
        if r.get("rejected"):
            rejections.append({"tool": tu.get("name", "?"), "label": label[:120]})
        elif r["is_error"]:
            errors.append({"tool": tu.get("name", "?"),
                           "label": label[:120],
                           "snippet": snippet})

    # --- duplicate identical commands ---
    bash_cmds = [tu["command"].strip()
                 for tu in tool_uses.values()
                 if tu["name"] == "Bash" and tu["command"].strip()]
    cmd_counts = Counter(bash_cmds)
    duplicates = {c: n for c, n in cmd_counts.items() if n > 1}

    # --- repeated file reads ---
    read_paths = [tu["input"].get("file_path", "")
                  for tu in tool_uses.values()
                  if tu["name"] == "Read" and tu["input"].get("file_path")]
    read_counts = Counter(read_paths)
    repeat_reads = {p: n for p, n in read_counts.items() if n > 1}

    # --- scriptability: same binary hammered raw many times ---
    binaries = Counter(_first_token(c) for c in bash_cmds if _first_token(c))
    clusters = {b: n for b, n in binaries.items()
                if n >= CONFIG["scriptability_threshold"]}

    # --- penalties ---
    err_pen = min(len(errors) * w["error_per_hit"], w["error_cap"])
    dup_extra = sum(n - 1 for n in duplicates.values())
    dup_pen = min(dup_extra * w["dup_command_per_extra"], w["dup_command_cap"])
    read_extra = sum(n - 1 for n in repeat_reads.values())
    read_pen = min(read_extra * w["repeat_read_per_extra"], w["repeat_read_cap"])
    script_pen = min(len(clusters) * w["scriptability_per_cluster"],
                     w["scriptability_cap"])
    reject_pen = min(len(rejections) * w["rejection_per_hit"], w["rejection_cap"])

    score = max(0, min(100, CONFIG["start_score"]
                       - err_pen - dup_pen - read_pen - script_pen - reject_pen))
    grade = next(g for floor, g in CONFIG["grades"] if score >= floor)

    return {
        "score": score,
        "grade": grade,
        "penalties": {
            "errors": err_pen,
            "duplicate_commands": dup_pen,
            "repeated_reads": read_pen,
            "scriptability": script_pen,
            "rejections": reject_pen,
        },
        "metrics": {
            "assistant_turns": parsed["assistant_turns"],
            "total_tool_calls": len(tool_uses),
            "bash_calls": len(bash_cmds),
            "tool_errors": len(errors),
            "user_rejections": len(rejections),
            "subagent_count": parsed.get("subagent_count", 0),
        },
        "detail": {
            "errors": errors,
            "rejections": rejections,
            "duplicates": duplicates,
            "repeat_reads": repeat_reads,
            "binary_clusters": clusters,
        },
    }


# ----------------------------------------------------------------------------
# Suggestions — the bit that feeds next-run skill tuning
# ----------------------------------------------------------------------------
def build_suggestions(scored: dict) -> list:
    """Turn scoring detail into actionable, human-readable suggestions."""
    d = scored["detail"]
    out = []
    if scored["metrics"]["tool_errors"]:
        out.append(
            f"{scored['metrics']['tool_errors']} tool error(s). The agent should "
            "capture and read failure output once, not retry blind. "
            "Top offender: " + (d["errors"][0]["label"] if d["errors"] else "n/a"))
    if scored["metrics"].get("user_rejections"):
        out.append(
            f"{scored['metrics']['user_rejections']} tool call(s) rejected by the user. "
            "Confirm intent before risky or irreversible actions. "
            "Top offender: " + (d["rejections"][0]["label"] if d["rejections"] else "n/a"))
    for cmd, n in sorted(d["duplicates"].items(), key=lambda x: -x[1])[:3]:
        out.append(f"Command run {n}x identically: `{cmd[:80]}`. "
                   "Cache the result or hoist it into a setup step.")
    for path, n in sorted(d["repeat_reads"].items(), key=lambda x: -x[1])[:3]:
        out.append(f"Re-read `{os.path.basename(path)}` {n}x. "
                   "Pin key files into context once up front.")
    for binary, n in sorted(d["binary_clusters"].items(), key=lambda x: -x[1])[:3]:
        out.append(f"`{binary}` called {n}x raw. When >"
                   f"{CONFIG['scriptability_threshold']} related shell steps, "
                   "write a reusable script or batch into one call.")
    if not out:
        out.append("Clean run. No structural waste detected.")
    return out


def headline(scored: dict) -> str:
    """Compact one-line summary, e.g. '84 B · 2 err · 1 dup'."""
    m = scored["metrics"]
    bits = [f"{scored['score']} {scored['grade']}"]
    if m["tool_errors"]:
        bits.append(f"{m['tool_errors']} err")
    if m.get("user_rejections"):
        bits.append(f"{m['user_rejections']} rej")
    dups = sum(n - 1 for n in scored["detail"]["duplicates"].values())
    if dups:
        bits.append(f"{dups} dup")
    return " · ".join(bits)


# ----------------------------------------------------------------------------
# Project label / first-prompt helpers (for leaderboard display)
# ----------------------------------------------------------------------------
_XML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")


def _clean_snippet(text: str) -> str:
    """Strip XML-ish tags and collapse whitespace for display."""
    text = _XML_TAG_RE.sub(" ", text)
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def humanize_project(raw: str) -> str:
    """Slug like '-Users-allannapier-code-pelai-api' -> 'pelai-api'."""
    name = raw
    if name.startswith("-"):
        name = name[1:]
    elif name.startswith("s-"):
        name = name[2:]
    if "-code-" in name:
        name = name.split("-code-", 1)[1]
    elif "-" in name:
        parts = name.split("-")
        name = "-".join(parts[-2:]) if len(parts) >= 2 else name
    return name or raw


def first_prompt(path: Path, max_lines: int = 80) -> str:
    """Cheap: first human text in the transcript, for labelling."""
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for _ in range(max_lines):
                line = fh.readline()
                if not line:
                    break
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("type") != "user":
                    continue
                content = (e.get("message") or {}).get("content")
                if isinstance(content, str):
                    text = _clean_snippet(content)
                    if text:
                        return text
                if isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "text":
                            text = _clean_snippet(str(b.get("text", "")))
                            if text:
                                return text
    except OSError:
        pass
    return ""


def evaluate_file(path: Path) -> Optional[dict]:
    """Parse + score a single transcript, returning a display-ready record."""
    try:
        scored = score_session(parse_transcript(path))
    except Exception:
        return None
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    raw_project = path.parent.name
    return {
        "session_id": path.stem,
        "project": raw_project,
        "project_label": humanize_project(raw_project),
        "mtime": mtime,
        "date": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M") if mtime else "",
        "prompt": first_prompt(path)[:120],
        "score": scored["score"],
        "grade": scored["grade"],
        "headline": headline(scored),
        "metrics": scored["metrics"],
        "penalties": scored["penalties"],
        "suggestions": build_suggestions(scored),
        "detail": scored["detail"],
    }
