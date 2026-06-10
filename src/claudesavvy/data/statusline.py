#!/usr/bin/env python3
"""Claude Code statusline: today's and month-to-date usage stats."""
import glob
import json
from datetime import datetime, timezone
from pathlib import Path

# ANSI colors
R = "\033[0m"       # reset
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
BLUE   = "\033[34m"
BOLD   = "\033[1m"
DIM    = "\033[2m"


# Per-MTok rates, synced with https://platform.claude.com/docs/en/about-claude/pricing
PRICING = {
    "fable":       {"input": 10.0, "output": 50.0, "cache_write": 12.50, "cache_read": 1.00},
    "opus":        {"input": 5.0,  "output": 25.0, "cache_write": 6.25,  "cache_read": 0.50},
    "opus_legacy": {"input": 15.0, "output": 75.0, "cache_write": 18.75, "cache_read": 1.50},
    "sonnet":      {"input": 3.0,  "output": 15.0, "cache_write": 3.75,  "cache_read": 0.30},
    "haiku":       {"input": 1.0,  "output": 5.0,  "cache_write": 1.25,  "cache_read": 0.10},
    "haiku_3_5":   {"input": 0.80, "output": 4.0,  "cache_write": 1.00,  "cache_read": 0.08},
}


def get_pricing(model: str) -> dict:
    if not model:
        return PRICING["sonnet"]
    m = model.lower()
    if "fable" in m or "mythos" in m:
        return PRICING["fable"]
    if "opus" in m:
        # Opus 4.1 and earlier (incl. Claude 3 Opus) use legacy $15/$75 rates;
        # "opus-4-2" matches the dated Opus 4 ID "claude-opus-4-20250514"
        if "opus-4-1" in m or "opus-4-2" in m or "opus-4-0" in m or "3-opus" in m:
            return PRICING["opus_legacy"]
        return PRICING["opus"]
    if "3-5-haiku" in m or "haiku-3-5" in m:
        return PRICING["haiku_3_5"]
    if "haiku" in m:
        return PRICING["haiku"]
    return PRICING["sonnet"]


def fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def main():
    # Match the dashboard's "today" / "this month" windows: local calendar
    # boundaries, converted to UTC for comparison with message timestamps.
    now_local = datetime.now().astimezone()
    cutoff_today = now_local.replace(
        hour=0, minute=0, second=0, microsecond=0
    ).astimezone(timezone.utc)
    cutoff_month = now_local.replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    ).astimezone(timezone.utc)
    claude_dir = Path.home() / ".claude"

    pattern = str(claude_dir / "projects" / "**" / "*.jsonl")
    files = glob.glob(pattern, recursive=True)

    # Track two windows in one pass
    day_cost = 0.0
    day_tokens = 0
    day_cache_write = 0
    day_cache_read = 0
    day_sessions: set = set()

    month_cost = 0.0

    for filepath in files:
        if Path(filepath).name == "history.jsonl":
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    ts_str = obj.get("timestamp", "")
                    if not ts_str:
                        continue
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    except ValueError:
                        continue

                    # Skip anything before month start
                    if ts < cutoff_month:
                        continue

                    msg = obj.get("message", {})
                    usage = msg.get("usage", {})
                    if not usage:
                        continue

                    inp = usage.get("input_tokens", 0) or 0
                    out = usage.get("output_tokens", 0) or 0
                    cw = usage.get("cache_creation_input_tokens", 0) or 0
                    cr = usage.get("cache_read_input_tokens", 0) or 0

                    if inp == 0 and out == 0 and cw == 0 and cr == 0:
                        continue

                    model = msg.get("model", "")
                    prices = get_pricing(model)
                    cost = (
                        inp * prices["input"]
                        + out * prices["output"]
                        + cw * prices["cache_write"]
                        + cr * prices["cache_read"]
                    ) / 1_000_000

                    month_cost += cost

                    if ts >= cutoff_today:
                        sid = obj.get("sessionId", "")
                        if sid:
                            day_sessions.add(sid)
                        day_cost += cost
                        day_tokens += inp + out + cw + cr
                        day_cache_write += cw
                        day_cache_read += cr

        except (OSError, PermissionError):
            continue

    if day_tokens == 0 and not day_sessions and month_cost == 0:
        print("no data", end="")
        return

    cache_denom = day_cache_write + day_cache_read
    if cache_denom > 0:
        cache_pct = int(day_cache_read / cache_denom * 100)
        # Green if >80%, yellow if >50%, else red
        cache_color = GREEN if cache_pct >= 80 else (YELLOW if cache_pct >= 50 else "\033[31m")
        cache_str = f"{DIM}cache{R} {cache_color}{BOLD}{cache_pct}%{R}"
    else:
        cache_str = f"{DIM}cache --{R}"

    sep = f" {DIM}|{R} "
    parts = [
        f"{DIM}today{R} {GREEN}{BOLD}${day_cost:.2f}{R}",
        f"{CYAN}{fmt_tokens(day_tokens)}{R} {DIM}tok{R}",
        cache_str,
        f"{BLUE}{len(day_sessions)}{R} {DIM}sess{R}",
        f"{DIM}month{R} {YELLOW}{BOLD}${month_cost:.2f}{R}",
    ]
    print(sep.join(parts), end="")


if __name__ == "__main__":
    main()
