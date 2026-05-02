"""Shared fixtures for quokka tests."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")


@pytest.fixture
def synthetic_harness(tmp_path: Path) -> dict:
    """Build a tmp_path with a fully-populated user home + project layout.

    Returns a dict with keys ``user_home`` and ``project`` so tests can pass
    them straight to :class:`quokka.scanner.ConfigScanner`.
    """
    user_home = tmp_path / "home"
    user_claude = user_home / ".claude"
    project = tmp_path / "proj"

    # ---- user-level skill -------------------------------------------------
    _write(
        user_claude / "skills" / "alpha" / "SKILL.md",
        """
        ---
        name: alpha
        description: Alpha skill for testing
        agents: [reviewer]
        ---
        # alpha
        Reusable alpha logic.
        """,
    )

    # ---- user-level sub-agent --------------------------------------------
    _write(
        user_claude / "agents" / "reviewer.md",
        """
        ---
        description: Reviews diffs
        model: sonnet
        tools: Bash, mcp__github__list_issues
        ---
        Review the user's pending diff.
        """,
    )

    # ---- user-level slash command ----------------------------------------
    _write(
        user_claude / "commands" / "ship.md",
        """
        ---
        description: Ship a release
        ---
        Run release pipeline.
        """,
    )

    # ---- user settings.json with hooks + MCP permissions -----------------
    settings = {
        "hooks": {
            "Stop": [
                {
                    "matcher": "",
                    "hooks": [{"type": "command", "command": "echo bye"}],
                }
            ],
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": "echo running"}],
                }
            ],
        },
        "permissions": {"allow": ["mcp__github__list_issues"]},
    }
    _write(user_claude / "settings.json", json.dumps(settings))

    # ---- user CLAUDE.md ---------------------------------------------------
    _write(
        user_claude / "CLAUDE.md",
        """
        # User notes
        Use /ship and ask reviewer for help.
        """,
    )

    # ---- plugin -----------------------------------------------------------
    plugin_root = user_claude / "plugins" / "store" / "demo-plugin"
    _write(
        plugin_root / ".claude-plugin" / "plugin.json",
        json.dumps(
            {
                "name": "demo-plugin",
                "description": "Demo plugin",
                "author": {"name": "ACME"},
                "keywords": ["demo"],
            }
        ),
    )
    _write(
        plugin_root / "skills" / "beta" / "SKILL.md",
        """
        ---
        description: Plugin-provided beta skill
        ---
        Beta details.
        """,
    )
    _write(
        plugin_root / "commands" / "deploy.md",
        """
        ---
        description: Deploy via plugin
        ---
        """,
    )
    _write(
        plugin_root / ".mcp.json",
        json.dumps(
            {
                "mcpServers": {
                    "github": {"command": "node", "args": ["server.js"]},
                }
            }
        ),
    )
    _write(
        plugin_root / "hooks" / "hooks.json",
        json.dumps(
            {
                "hooks": {
                    "PostToolUse": [
                        {
                            "matcher": "Edit",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "${CLAUDE_PLUGIN_ROOT}/notify.sh",
                                }
                            ],
                        }
                    ]
                }
            }
        ),
    )

    # ---- installed plugins manifest --------------------------------------
    _write(
        user_claude / "plugins" / "installed_plugins.json",
        json.dumps(
            {
                "plugins": {
                    "demo-plugin": [
                        {"version": "1.0.0", "installPath": str(plugin_root)}
                    ]
                }
            }
        ),
    )

    # ---- project ----------------------------------------------------------
    project.mkdir()
    _write(
        project / "CLAUDE.md",
        """
        # Project rules
        Defer to /ship for releases. The alpha workflow is canonical.
        Avoid hitting /docs/foo style URLs in examples.
        ```
        /ship inside-fence
        ```
        """,
    )
    _write(
        project / "AGENTS.md",
        """
        # Agents
        The reviewer agent owns all PR comments.
        """,
    )
    _write(
        project / ".claude" / "settings.local.json",
        json.dumps(
            {
                "hooks": {
                    "UserPromptSubmit": [
                        {
                            "matcher": "",
                            "hooks": [{"type": "command", "command": "echo prompt"}],
                        }
                    ]
                }
            }
        ),
    )

    return {"user_home": user_home, "project": project}
