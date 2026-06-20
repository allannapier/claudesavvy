"""Demo data generator for ClaudeSavvy demo mode.

Generates a realistic fake ~/.claude/ directory in a temp location so the
full parsing/analysis pipeline runs against sample data without needing an
actual Claude Code installation.
"""

import json
import random
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from ..utils.paths import ClaudeDataPaths

# ---------------------------------------------------------------------------
# Data tables
# ---------------------------------------------------------------------------

_MODELS = [
    ("claude-sonnet-4-5-20251022", 0.70),
    ("claude-haiku-4-5-20251001", 0.20),
    ("claude-opus-4-5-20251101", 0.10),
]

_PROJECTS = [
    {
        "cwd": "/home/demo/claudesavvy",
        "dir": "-home-demo-claudesavvy",
        "n_sessions": 8,
        "scale": 1.0,
    },
    {
        "cwd": "/home/demo/my-saas-app",
        "dir": "-home-demo-my-saas-app",
        "n_sessions": 5,
        "scale": 0.80,
    },
    {
        "cwd": "/home/demo/data-pipeline",
        "dir": "-home-demo-data-pipeline",
        "n_sessions": 3,
        "scale": 0.60,
    },
    {
        "cwd": "/home/demo/blog",
        "dir": "-home-demo-blog",
        "n_sessions": 2,
        "scale": 0.40,
    },
]

# ---------------------------------------------------------------------------
# Tool input pools — Bash and Read use shuffled no-repeat queues so base
# sessions never produce natural duplicates/scriptability clusters.
# Each bash pool has diverse binaries (≤3 uses of any single binary) and
# excludes "python" and "pytest" (reserved for scriptability injection).
# ---------------------------------------------------------------------------

_BASH_POOLS: dict = {
    "/home/demo/claudesavvy": [
        "git diff HEAD --stat",
        "git log --oneline -10",
        "git status",
        "ruff check src/",
        "mypy src/ --ignore-missing-imports",
        "black --check src/",
        "pre-commit run --all-files",
        "ls src/claudesavvy/ -la",
        "cat pyproject.toml",
        "wc -l src/claudesavvy/**/*.py",
        "find src/ -name '*.py' | head -20",
        "git show --stat HEAD",
    ],
    "/home/demo/my-saas-app": [
        "npm test",
        "npm run build",
        "npm run lint",
        "npm run type-check",
        "git status",
        "git diff --stat",
        "git log --oneline -5",
        "npx tsc --noEmit",
        "npx eslint src/ --ext .ts,.tsx",
        "ls src/components/ -la",
    ],
    "/home/demo/data-pipeline": [
        "bash tests/run_tests.sh",
        "git diff pipeline/",
        "git log --oneline -5 pipeline/",
        "cat pipeline/config.yaml",
        "ls pipeline/ -la",
        "make validate",
        "make test",
        "wc -l pipeline/*.py",
        "diff pipeline/config.yaml pipeline/config.yaml.bak",
    ],
    "/home/demo/blog": [
        "npm run build",
        "npm run serve",
        "npm run preview",
        "git diff --stat",
        "git log --oneline -5",
        "ls content/posts/ -la",
        "wc -w content/posts/*.md",
        "cat config.toml",
    ],
}

_READ_POOLS: dict = {
    "/home/demo/claudesavvy": [
        "{c}/src/claudesavvy/web/app.py",
        "{c}/src/claudesavvy/cli.py",
        "{c}/src/claudesavvy/data/demo.py",
        "{c}/src/claudesavvy/analyzers/harness_quality.py",
        "{c}/src/claudesavvy/web/routes/dashboard.py",
        "{c}/src/claudesavvy/web/services/dashboard_service.py",
        "{c}/pyproject.toml",
        "{c}/src/claudesavvy/parsers/sessions.py",
        "{c}/src/claudesavvy/utils/paths.py",
        "{c}/src/claudesavvy/web/templates/pages/harness.html",
        "{c}/scripts/render_static.py",
        "{c}/README.md",
    ],
    "/home/demo/my-saas-app": [
        "{c}/src/auth/views.py",
        "{c}/src/models/user.py",
        "{c}/src/api/endpoints.py",
        "{c}/src/components/Dashboard.tsx",
        "{c}/src/utils/helpers.ts",
        "{c}/package.json",
        "{c}/src/auth/middleware.py",
        "{c}/src/models/session.py",
        "{c}/src/api/auth.py",
        "{c}/tsconfig.json",
    ],
    "/home/demo/data-pipeline": [
        "{c}/pipeline/transform.py",
        "{c}/pipeline/ingest.py",
        "{c}/pipeline/config.yaml",
        "{c}/pipeline/run.py",
        "{c}/pipeline/validate.py",
        "{c}/pipeline/schema.json",
        "{c}/pipeline/utils.py",
        "{c}/tests/test_transform.py",
    ],
    "/home/demo/blog": [
        "{c}/content/posts/intro.md",
        "{c}/content/posts/tutorial.md",
        "{c}/content/posts/faq.md",
        "{c}/config.toml",
        "{c}/themes/default/style.css",
        "{c}/themes/default/base.html",
        "{c}/content/posts/release-notes.md",
        "{c}/layouts/index.html",
    ],
}

# Non-Read/Bash tool factories — these use rng for variety but aren't tracked
# by the harness scorer, so natural duplicates don't matter.
def _rtool(inputs: list[dict]):
    return lambda c, r: {
        k: (v.replace("{c}", c) if isinstance(v, str) else v)
        for k, v in r.choice(inputs).items()
    }


_TOOLS = {
    "/home/demo/claudesavvy": [
        ("Read", 0.30, None),   # handled by _READ_POOLS queue
        ("Edit", 0.20, _rtool([
            {"file_path": "{c}/src/claudesavvy/cli.py", "old_string": "# old", "new_string": "# new"},
            {"file_path": "{c}/src/claudesavvy/data/demo.py", "old_string": "# old", "new_string": "# new"},
            {"file_path": "{c}/src/claudesavvy/web/routes/dashboard.py", "old_string": "# old", "new_string": "# new"},
            {"file_path": "{c}/src/claudesavvy/web/services/dashboard_service.py", "old_string": "# old", "new_string": "# new"},
        ])),
        ("Bash", 0.15, None),   # handled by _BASH_POOLS queue
        ("Glob", 0.10, _rtool([
            {"pattern": "**/*.py"},
            {"pattern": "src/**/*.html"},
            {"pattern": "tests/**/*.py"},
        ])),
        ("Grep", 0.08, _rtool([
            {"pattern": "def create_app", "path": "{c}"},
            {"pattern": "class Session", "path": "{c}"},
            {"pattern": "import flask", "path": "{c}"},
            {"pattern": "total_tokens", "path": "{c}"},
        ])),
        ("Write", 0.05, _rtool([
            {"file_path": "{c}/src/claudesavvy/data/demo.py", "content": "..."},
            {"file_path": "{c}/docs/demo/index.html", "content": "..."},
        ])),
        ("mcp__github__get_file_contents", 0.05, lambda c, r: {"owner": "allannapier", "repo": "claudesavvy", "path": "README.md"}),
        ("mcp__github__list_issues",       0.04, lambda c, r: {"owner": "allannapier", "repo": "claudesavvy"}),
        ("Task", 0.03, lambda c, r: {"subagent_type": "Explore", "description": "Find all route handlers"}),
    ],
    "/home/demo/my-saas-app": [
        ("Read", 0.25, None),
        ("Edit", 0.20, _rtool([
            {"file_path": "{c}/src/auth/views.py", "old_string": "# old", "new_string": "# new"},
            {"file_path": "{c}/src/models/user.py", "old_string": "# old", "new_string": "# new"},
            {"file_path": "{c}/src/api/endpoints.py", "old_string": "# old", "new_string": "# new"},
        ])),
        ("Bash", 0.18, None),
        ("Write", 0.12, _rtool([
            {"file_path": "{c}/src/components/Button.tsx", "content": "..."},
            {"file_path": "{c}/src/components/Modal.tsx", "content": "..."},
            {"file_path": "{c}/src/utils/helpers.ts", "content": "..."},
        ])),
        ("Glob", 0.08, _rtool([
            {"pattern": "src/**/*.tsx"},
            {"pattern": "src/**/*.ts"},
            {"pattern": "**/*.test.ts"},
        ])),
        ("mcp__github__list_issues",        0.07, lambda c, r: {"owner": "demo", "repo": "my-saas-app"}),
        ("Task",                             0.05, lambda c, r: {"subagent_type": "general-purpose", "description": "Research auth patterns"}),
        ("mcp__Google_Drive__search_files",  0.05, lambda c, r: {"query": "design mockups"}),
    ],
    "/home/demo/data-pipeline": [
        ("Read", 0.30, None),
        ("Bash", 0.30, None),
        ("Edit", 0.20, _rtool([
            {"file_path": "{c}/pipeline/transform.py", "old_string": "# old", "new_string": "# new"},
            {"file_path": "{c}/pipeline/ingest.py", "old_string": "# old", "new_string": "# new"},
        ])),
        ("Write", 0.10, _rtool([
            {"file_path": "{c}/pipeline/config.yaml", "content": "..."},
            {"file_path": "{c}/pipeline/schema.json", "content": "..."},
        ])),
        ("Task", 0.10, lambda c, r: {"subagent_type": "code-reviewer", "description": "Review pipeline logic"}),
    ],
    "/home/demo/blog": [
        ("Read", 0.35, None),
        ("Write", 0.30, _rtool([
            {"file_path": "{c}/content/posts/new-post.md", "content": "..."},
            {"file_path": "{c}/content/posts/update.md", "content": "..."},
            {"file_path": "{c}/content/posts/announcement.md", "content": "..."},
        ])),
        ("Edit", 0.20, _rtool([
            {"file_path": "{c}/content/posts/intro.md", "old_string": "# old", "new_string": "# new"},
            {"file_path": "{c}/config.toml", "old_string": "# old", "new_string": "# new"},
        ])),
        ("Bash", 0.15, None),
    ],
}

_PROMPTS = {
    "/home/demo/claudesavvy": [
        "add a --demo flag that loads sample data",
        "create a static HTML renderer for GitHub Pages",
        "fix the chart data not refreshing on time filter change",
        "add MCP integration usage to the features page",
        "improve project breakdown table sorting",
        "add CSV export functionality",
        "fix the cache hit rate calculation",
        "add pagination to the conversations view",
        "optimize the session parser for large files",
        "write unit tests for the token analyzer",
        "update pricing constants for new Claude models",
        "add a demo banner to the static output",
    ],
    "/home/demo/my-saas-app": [
        "implement user authentication with JWT tokens",
        "add Stripe payment integration",
        "create the dashboard UI components",
        "fix the login form validation errors",
        "add email verification flow",
        "implement API rate limiting",
        "add Google OAuth login option",
        "write tests for the auth module",
        "set up CI/CD pipeline",
    ],
    "/home/demo/data-pipeline": [
        "add error handling to the ETL pipeline",
        "optimize the data transformation step",
        "add monitoring and alerting",
        "fix the date parsing bug in ingestion",
        "add support for JSON input format",
        "write documentation for the pipeline",
    ],
    "/home/demo/blog": [
        "write an intro post about the project",
        "add syntax highlighting to code blocks",
        "fix the RSS feed generation",
        "add a search feature",
    ],
}

_MCP_SERVERS = ["github", "Google_Drive", "Vercel"]

# ---------------------------------------------------------------------------
# Harness quality profiles — control the grade distribution on the harness page
# ---------------------------------------------------------------------------

# Each profile name maps to rates/counts that drive the harness scorer.
# Tuned so the weighted scoring formula produces the expected letter grade.
_QUALITY_PROFILES: dict = {
    # error_rate / rejection_rate: fraction of base tool calls that error/get rejected
    # extra_dup_bash: extra turns repeating an identical bash command (triggers duplicate_commands)
    # extra_rep_reads: extra turns re-reading the same file (triggers repeated_reads)
    # script_binary: binary to call 5× with varied args (triggers scriptability cluster)
    "A": {"error_rate": 0.00, "rejection_rate": 0.00, "extra_dup_bash": 0, "extra_rep_reads": 0, "script_binary": None},
    "B": {"error_rate": 0.10, "rejection_rate": 0.02, "extra_dup_bash": 1, "extra_rep_reads": 1, "script_binary": None},
    "C": {"error_rate": 0.12, "rejection_rate": 0.06, "extra_dup_bash": 2, "extra_rep_reads": 2, "script_binary": "pytest"},
    "D": {"error_rate": 0.18, "rejection_rate": 0.10, "extra_dup_bash": 3, "extra_rep_reads": 3, "script_binary": "python"},
    "F": {"error_rate": 0.40, "rejection_rate": 0.30, "extra_dup_bash": 5, "extra_rep_reads": 4, "script_binary": "python"},
}

# Per-project session quality assignments (index 0 = today's session, most visible).
_PROJECT_QUALITY: dict = {
    "-home-demo-claudesavvy":  ["C", "A", "B", "D", "B", "A", "C", "B"],
    "-home-demo-my-saas-app":  ["D", "B", "C", "A", "C"],
    "-home-demo-data-pipeline": ["F", "C", "B"],
    "-home-demo-blog":          ["B", "A"],
}

# Varied bash args for scriptability injection so calls are not flagged as duplicates.
_SCRIPT_ARGS: dict = {
    "pytest": [
        "tests/test_auth.py -v", "tests/test_api.py -v", "tests/test_models.py -v",
        "tests/test_utils.py -v", "tests/ -x --tb=short",
    ],
    "python": [
        "scripts/run.py", "scripts/process.py", "scripts/analyze.py",
        "scripts/report.py", "scripts/clean.py",
    ],
}

_SETTINGS = {
    "mcpServers": {
        "github": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
        },
        "Google_Drive": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@google/mcp-server-drive"],
        },
        "Vercel": {
            "type": "stdio",
            "command": "npx",
            "args": ["@vercel/mcp-adapter"],
        },
    },
    "permissions": {
        "allow": ["Bash(npm:*)", "Bash(python:*)", "Bash(git:*)"],
        "deny": [],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick(rng: random.Random, choices):
    """Weighted pick from [(value, weight, ...)] — returns first element."""
    total = sum(row[1] for row in choices)
    r = rng.random() * total
    for row in choices:
        r -= row[1]
        if r <= 0:
            return row
    return choices[-1]


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _tool_result_text(rng: random.Random, tool_name: str, tool_input: dict) -> str:
    """Generate plausible success output for a tool call."""
    if tool_name == "Read":
        fp = tool_input.get("file_path", "file.py")
        name = fp.split("/")[-1]
        return f"     1\t# {name}\n     2\tdef main():\n     3\t    pass\n"
    if tool_name == "Bash":
        cmd = tool_input.get("command", "")
        if any(t in cmd for t in ("pytest", "test", "npm test")):
            n = rng.randint(3, 15)
            secs = rng.uniform(0.5, 5.0)
            return f"========================= {n} passed in {secs:.2f}s ========================="
        if "git" in cmd:
            return "On branch main\nYour branch is up to date with 'origin/main'.\n"
        if any(t in cmd for t in ("pip", "npm install")):
            return "Successfully installed packages\n"
        return ""
    if tool_name == "Edit":
        return "The file has been edited successfully."
    if tool_name == "Write":
        return "File written successfully."
    if tool_name == "Glob":
        return "src/main.py\nsrc/utils.py\nsrc/models.py\n"
    if tool_name == "Grep":
        return "src/main.py:15:def create_app():\n"
    if "mcp__github__" in tool_name:
        return '{"total_count": 3, "items": []}'
    if "mcp__" in tool_name:
        return '{"result": "success"}'
    if tool_name == "Task":
        return "Task completed successfully."
    return "OK"


def _tool_error_text(rng: random.Random, tool_name: str, tool_input: dict) -> str:
    """Generate realistic error output for a tool call."""
    if tool_name == "Bash":
        cmd = tool_input.get("command", "").strip()
        binary = cmd.split()[0] if cmd.split() else "command"
        choices = [
            f"bash: {binary}: command not found",
            (
                "Traceback (most recent call last):\n"
                "  File \"main.py\", line 42, in run\n"
                "    result = process(data)\n"
                "TypeError: expected str, got int"
            ),
            "FAILED tests/test_main.py::test_process - AssertionError: assert 42 == 0\n1 failed in 0.34s",
            f"No such file or directory: '{cmd.split()[-1]}'",
        ]
        return rng.choice(choices)
    if tool_name == "Read":
        fp = tool_input.get("file_path", "file.py")
        return f"Error: No such file or directory: '{fp}'"
    if tool_name == "Edit":
        return "Error: old_string not found in file. The content may have changed."
    return "Error: operation failed unexpectedly"


def _generate_session(
    rng: random.Random,
    cwd: str,
    session_id: str,
    start: datetime,
    model: str,
    n_turns: int,
    scale: float,
    quality: Optional[dict] = None,
) -> list[str]:
    """Return JSONL lines for one session.

    quality dict keys: error_rate, rejection_rate, extra_dup_bash,
    extra_rep_reads, script_binary — see _QUALITY_PROFILES.
    """
    quality = quality or {}
    error_rate: float = quality.get("error_rate", 0.0)
    rejection_rate: float = quality.get("rejection_rate", 0.0)
    extra_dup_bash: int = quality.get("extra_dup_bash", 0)
    extra_rep_reads: int = quality.get("extra_rep_reads", 0)
    script_binary: Optional[str] = quality.get("script_binary")

    lines: list[str] = []
    prompts = _PROMPTS.get(cwd, _PROMPTS["/home/demo/claudesavvy"])
    tools = _TOOLS.get(cwd, _TOOLS["/home/demo/claudesavvy"])

    # Shuffled no-repeat queues for Bash and Read — prevents natural duplicates
    # that would distort harness scores for clean (A) sessions.
    bash_pool = list(_BASH_POOLS.get(cwd, _BASH_POOLS["/home/demo/claudesavvy"]))
    read_pool = list(_READ_POOLS.get(cwd, _READ_POOLS["/home/demo/claudesavvy"]))
    rng.shuffle(bash_pool)
    rng.shuffle(read_pool)
    bash_qi = 0
    read_qi = 0

    def _next_bash() -> str:
        nonlocal bash_qi
        if bash_qi >= len(bash_pool):
            rng.shuffle(bash_pool)
            bash_qi = 0
        cmd = bash_pool[bash_qi]
        bash_qi += 1
        return cmd

    def _next_read() -> str:
        nonlocal read_qi
        if read_qi >= len(read_pool):
            rng.shuffle(read_pool)
            read_qi = 0
        fp = read_pool[read_qi].replace("{c}", cwd)
        read_qi += 1
        return fp

    cache_pool = 0
    tc_counter = 0  # unique tool-call ID counter within this session
    seen_bash_cmds: list[str] = []
    seen_read_paths: list[str] = []

    def make_tc_id() -> str:
        nonlocal tc_counter
        tc_counter += 1
        return f"tc_{session_id}_{tc_counter:04d}"

    for i in range(n_turns):
        ts = start + timedelta(seconds=i * 90 + rng.randint(-15, 30))

        # User prompt
        lines.append(json.dumps({
            "type": "user",
            "timestamp": _ts(ts),
            "sessionId": session_id,
            "cwd": cwd,
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": prompts[i % len(prompts)]}],
            },
        }))

        ts_resp = ts + timedelta(seconds=rng.randint(4, 20))

        # Token model: cache fills up over the session
        progress = i / max(n_turns - 1, 1)
        base_fresh = int(rng.randint(2000, 7000) * scale * (1 - progress * 0.7))
        cache_creation = int(rng.randint(
            int(15000 * scale) if i == 0 else int(1000 * scale),
            int(25000 * scale) if i == 0 else int(8000 * scale),
        ) * (1 - progress * 0.85))
        cache_read = int(cache_pool * rng.uniform(0.85, 1.0))
        output = int(rng.randint(300, 2500) * scale)
        cache_pool += cache_creation

        # Pick 1-3 tools
        n_tools = rng.choices([1, 2, 3], weights=[0.50, 0.35, 0.15])[0]
        tool_uses_this_turn: list[dict] = []
        for _ in range(n_tools):
            row = _pick(rng, tools)
            tool_name, _, input_fn = row[0], row[1], row[2]
            tc_id = make_tc_id()
            # Read and Bash use no-repeat queues; others use their factory
            if tool_name == "Read":
                fp = _next_read()
                tool_input = {"file_path": fp}
                seen_read_paths.append(fp)
            elif tool_name == "Bash":
                cmd = _next_bash()
                tool_input = {"command": cmd}
                seen_bash_cmds.append(cmd)
            else:
                tool_input = input_fn(cwd, rng) if input_fn else {}
            tool_uses_this_turn.append({
                "type": "tool_use",
                "id": tc_id,
                "name": tool_name,
                "input": tool_input,
            })

        # Assistant message with tool calls
        lines.append(json.dumps({
            "type": "assistant",
            "timestamp": _ts(ts_resp),
            "sessionId": session_id,
            "cwd": cwd,
            "message": {
                "role": "assistant",
                "model": model,
                "content": tool_uses_this_turn,
                "usage": {
                    "input_tokens": max(100, base_fresh),
                    "output_tokens": max(50, output),
                    "cache_creation_input_tokens": max(0, cache_creation),
                    "cache_read_input_tokens": max(0, cache_read),
                },
            },
        }))

        ts_result = ts_resp + timedelta(seconds=rng.randint(1, 8))

        # User message with tool results
        result_content: list[dict] = []
        for tu in tool_uses_this_turn:
            is_rej = rng.random() < rejection_rate
            is_err = (not is_rej) and rng.random() < error_rate
            if is_rej:
                content_text = "The user doesn't want to proceed with this action."
                err_flag = False
            elif is_err:
                content_text = _tool_error_text(rng, tu["name"], tu.get("input", {}))
                err_flag = True
            else:
                content_text = _tool_result_text(rng, tu["name"], tu.get("input", {}))
                err_flag = False
            result_content.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": content_text,
                "is_error": err_flag,
            })

        lines.append(json.dumps({
            "type": "user",
            "timestamp": _ts(ts_result),
            "sessionId": session_id,
            "cwd": cwd,
            "message": {"role": "user", "content": result_content},
        }))

    # -----------------------------------------------------------------------
    # Inject imperfection turns at the end (duplicates, repeated reads,
    # scriptability) — the harness scorer counts totals not position.
    # -----------------------------------------------------------------------
    last_ts = start + timedelta(seconds=n_turns * 90 + 60)

    def _inject_turn(tool_name: str, tool_input: dict, offset_secs: int) -> None:
        tc_id = make_tc_id()
        ts_imp = last_ts + timedelta(seconds=offset_secs)
        lines.append(json.dumps({
            "type": "assistant",
            "timestamp": _ts(ts_imp),
            "sessionId": session_id,
            "cwd": cwd,
            "message": {
                "role": "assistant",
                "model": model,
                "content": [{"type": "tool_use", "id": tc_id, "name": tool_name, "input": tool_input}],
                "usage": {"input_tokens": 400, "output_tokens": 80,
                          "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            },
        }))
        lines.append(json.dumps({
            "type": "user",
            "timestamp": _ts(ts_imp + timedelta(seconds=3)),
            "sessionId": session_id,
            "cwd": cwd,
            "message": {"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": tc_id,
                "content": _tool_result_text(rng, tool_name, tool_input),
                "is_error": False,
            }]},
        }))

    offset = 0

    # Duplicate bash: repeat the same command identically (triggers duplicate_commands)
    if extra_dup_bash > 0 and seen_bash_cmds:
        for k in range(extra_dup_bash):
            cmd = seen_bash_cmds[k % len(seen_bash_cmds)]
            _inject_turn("Bash", {"command": cmd}, offset)
            offset += 30

    # Repeated reads: re-read the same file (triggers repeated_reads)
    if extra_rep_reads > 0 and seen_read_paths:
        for k in range(extra_rep_reads):
            fp = seen_read_paths[k % len(seen_read_paths)]
            _inject_turn("Read", {"file_path": fp}, offset)
            offset += 25

    # Scriptability: call the same binary ≥5× with varied args (triggers scriptability cluster)
    if script_binary:
        args_list = _SCRIPT_ARGS.get(script_binary, [f"{script_binary} --help"])
        for k in range(5):
            cmd = f"{script_binary} {args_list[k % len(args_list)]}"
            _inject_turn("Bash", {"command": cmd}, offset)
            offset += 20

    return lines


def _write_debug_logs(base: Path, now: datetime, rng: random.Random) -> None:
    """Write fake MCP debug log text files."""
    for i, server in enumerate(_MCP_SERVERS):
        log_lines = []
        for day in range(1, 8):
            dt = now - timedelta(days=day, hours=rng.randint(8, 18))
            log_lines.append(f'[{_ts(dt)}] MCP server "{server}": Initializing')
            log_lines.append(f'[{_ts(dt)}] MCP server "{server}": connected')
            for _ in range(rng.randint(5, 20)):
                dt += timedelta(minutes=rng.randint(1, 15))
                tool = f"mcp__{server}__list_tools"
                log_lines.append(f'[{_ts(dt)}] Tool call: {tool}')
        log_file = base / "debug" / f"mcp-session-{i + 1:04d}.txt"
        log_file.write_text("\n".join(log_lines) + "\n")


def _write_skills(base: Path) -> None:
    """Write a few fake installed skills."""
    skills_dir = base / "skills"
    for name in ["code-review", "security-review", "run"]:
        skill_dir = skills_dir / name
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text(f"# {name}\nA demo skill.\n")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_demo_data() -> ClaudeDataPaths:
    """Generate fake demo data into a temp directory.

    Returns a ClaudeDataPaths pointing at the generated directory. The
    directory is in /tmp and will be cleaned up by the OS eventually.
    """
    rng = random.Random(42)
    now = datetime.now(timezone.utc)

    tmpdir = tempfile.mkdtemp(prefix="claudesavvy-demo-")
    base = Path(tmpdir)

    for d in ["projects", "debug", "file-history", "skills"]:
        (base / d).mkdir()

    (base / "settings.json").write_text(json.dumps(_SETTINGS, indent=2))

    _write_debug_logs(base, now, rng)
    _write_skills(base)

    history_entries: list[dict] = []

    session_counter = 0
    for proj in _PROJECTS:
        proj_dir = base / "projects" / proj["dir"]
        proj_dir.mkdir()

        n_sessions: int = proj["n_sessions"]
        scale: float = proj["scale"]
        cwd: str = proj["cwd"]

        # Spread sessions: one always today, rest over last 30 days
        historical = sorted(rng.sample(range(1, 30), n_sessions - 1), reverse=True)
        day_offsets = [0] + historical

        quality_list = _PROJECT_QUALITY.get(proj["dir"], [])

        for idx, day_offset in enumerate(day_offsets):
            session_counter += 1
            session_id = f"demo-session-{session_counter:04d}"

            session_start = now - timedelta(
                days=day_offset,
                hours=rng.randint(7, 21),
                minutes=rng.randint(0, 59),
            )

            model_row = _pick(rng, [(m, w) for m, w in _MODELS])
            model = model_row[0]
            n_turns = rng.randint(12, 24)

            quality_name = quality_list[idx] if idx < len(quality_list) else "A"
            quality = _QUALITY_PROFILES[quality_name]

            lines = _generate_session(rng, cwd, session_id, session_start, model, n_turns, scale, quality)
            (proj_dir / f"{session_id}.jsonl").write_text("\n".join(lines) + "\n")

            # Add history entries for this session
            prompts = _PROMPTS.get(cwd, _PROMPTS["/home/demo/claudesavvy"])
            n_commands = rng.randint(6, 14)
            for j in range(n_commands):
                cmd_dt = session_start + timedelta(minutes=j * 3 + rng.randint(0, 2))
                history_entries.append({
                    "display": prompts[j % len(prompts)],
                    "timestamp": int(cmd_dt.timestamp() * 1000),
                    "project": cwd,
                    "pastedContents": {},
                })

    history_entries.sort(key=lambda e: e["timestamp"])
    (base / "history.jsonl").write_text(
        "\n".join(json.dumps(e) for e in history_entries) + "\n"
    )

    return ClaudeDataPaths(base_dir=base)
