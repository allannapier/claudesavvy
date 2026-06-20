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

_TOOLS = {
    "/home/demo/claudesavvy": [
        ("Read",    0.30, lambda c: {"file_path": f"{c}/src/claudesavvy/web/app.py"}),
        ("Edit",    0.20, lambda c: {"file_path": f"{c}/src/claudesavvy/cli.py", "old_string": "# old", "new_string": "# new"}),
        ("Bash",    0.15, lambda c: {"command": "python -m pytest tests/ -v"}),
        ("Glob",    0.10, lambda c: {"pattern": "**/*.py"}),
        ("Grep",    0.08, lambda c: {"pattern": "def create_app", "path": c}),
        ("Write",   0.05, lambda c: {"file_path": f"{c}/src/claudesavvy/data/demo.py", "content": "..."}),
        ("mcp__github__get_file_contents", 0.05, lambda c: {"owner": "allannapier", "repo": "claudesavvy", "path": "README.md"}),
        ("mcp__github__list_issues",       0.04, lambda c: {"owner": "allannapier", "repo": "claudesavvy"}),
        ("Task",    0.03, lambda c: {"subagent_type": "Explore", "description": "Find all route handlers"}),
    ],
    "/home/demo/my-saas-app": [
        ("Read",    0.25, lambda c: {"file_path": f"{c}/src/auth/views.py"}),
        ("Edit",    0.20, lambda c: {"file_path": f"{c}/src/auth/views.py", "old_string": "# old", "new_string": "# new"}),
        ("Bash",    0.18, lambda c: {"command": "npm test"}),
        ("Write",   0.12, lambda c: {"file_path": f"{c}/src/components/Button.tsx", "content": "..."}),
        ("Glob",    0.08, lambda c: {"pattern": "src/**/*.tsx"}),
        ("mcp__github__list_issues",       0.07, lambda c: {"owner": "demo", "repo": "my-saas-app"}),
        ("Task",    0.05, lambda c: {"subagent_type": "general-purpose", "description": "Research auth patterns"}),
        ("mcp__Google_Drive__search_files", 0.05, lambda c: {"query": "design mockups"}),
    ],
    "/home/demo/data-pipeline": [
        ("Read",    0.30, lambda c: {"file_path": f"{c}/pipeline/transform.py"}),
        ("Bash",    0.30, lambda c: {"command": "python pipeline/run.py --dry-run"}),
        ("Edit",    0.20, lambda c: {"file_path": f"{c}/pipeline/transform.py", "old_string": "# old", "new_string": "# new"}),
        ("Write",   0.10, lambda c: {"file_path": f"{c}/pipeline/config.yaml", "content": "..."}),
        ("Task",    0.10, lambda c: {"subagent_type": "code-reviewer", "description": "Review pipeline logic"}),
    ],
    "/home/demo/blog": [
        ("Read",    0.35, lambda c: {"file_path": f"{c}/content/posts/intro.md"}),
        ("Write",   0.30, lambda c: {"file_path": f"{c}/content/posts/new-post.md", "content": "..."}),
        ("Edit",    0.20, lambda c: {"file_path": f"{c}/content/posts/intro.md", "old_string": "# old", "new_string": "# new"}),
        ("Bash",    0.15, lambda c: {"command": "npm run build"}),
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


def _generate_session(
    rng: random.Random,
    cwd: str,
    session_id: str,
    start: datetime,
    model: str,
    n_turns: int,
    scale: float,
) -> list[str]:
    """Return JSONL lines for one session."""
    lines: list[str] = []
    prompts = _PROMPTS.get(cwd, _PROMPTS["/home/demo/claudesavvy"])
    tools = _TOOLS.get(cwd, _TOOLS["/home/demo/claudesavvy"])

    # Cache accumulates across turns (sum of previous writes)
    cache_pool = 0

    for i in range(n_turns):
        ts = start + timedelta(seconds=i * 90 + rng.randint(-15, 30))

        # User turn
        prompt = prompts[i % len(prompts)]
        lines.append(json.dumps({
            "type": "user",
            "timestamp": _ts(ts),
            "sessionId": session_id,
            "cwd": cwd,
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
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
        content = []
        for _ in range(n_tools):
            row = _pick(rng, tools)
            tool_name, _, input_fn = row[0], row[1], row[2]
            content.append({
                "type": "tool_use",
                "name": tool_name,
                "input": input_fn(cwd),
            })

        lines.append(json.dumps({
            "type": "assistant",
            "timestamp": _ts(ts_resp),
            "sessionId": session_id,
            "cwd": cwd,
            "message": {
                "role": "assistant",
                "model": model,
                "content": content,
                "usage": {
                    "input_tokens": max(100, base_fresh),
                    "output_tokens": max(50, output),
                    "cache_creation_input_tokens": max(0, cache_creation),
                    "cache_read_input_tokens": max(0, cache_read),
                },
            },
        }))

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

        for day_offset in day_offsets:
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

            lines = _generate_session(rng, cwd, session_id, session_start, model, n_turns, scale)
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
