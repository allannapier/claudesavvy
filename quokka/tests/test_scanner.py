"""Scanner discovery tests."""

from __future__ import annotations

from pathlib import Path

from quokka.scanner import (
    SOURCE_PLUGIN,
    SOURCE_PROJECT,
    SOURCE_USER,
    ConfigScanner,
    parse_frontmatter,
)


def test_parse_frontmatter_basic() -> None:
    fm, body = parse_frontmatter("---\ndescription: hi\nmodel: sonnet\n---\nbody")
    assert fm == {"description": "hi", "model": "sonnet"}
    assert body == "body"


def test_parse_frontmatter_missing() -> None:
    fm, body = parse_frontmatter("just a body")
    assert fm == {}
    assert body == "just a body"


def test_scan_finds_user_and_project_artifacts(synthetic_harness: dict) -> None:
    scanner = ConfigScanner(user_home=synthetic_harness["user_home"])
    data = scanner.scan(synthetic_harness["project"])

    skill_names = {(s.name, s.source) for s in data.skills}
    assert ("alpha", SOURCE_USER) in skill_names
    assert ("beta", SOURCE_PLUGIN) in skill_names

    agent_names = {(a.name, a.source) for a in data.agents}
    assert ("reviewer", SOURCE_USER) in agent_names

    command_names = {(c.name, c.source) for c in data.commands}
    assert ("ship", SOURCE_USER) in command_names
    assert ("deploy", SOURCE_PLUGIN) in command_names

    plugin_names = {p.name for p in data.plugins}
    assert "demo-plugin" in plugin_names

    mcp_names = {m.name for m in data.mcps}
    assert "github" in mcp_names

    hook_types = {(h.hook_type, h.source) for h in data.hooks}
    assert ("Stop", SOURCE_USER) in hook_types
    assert ("PreToolUse", SOURCE_USER) in hook_types
    assert ("PostToolUse", SOURCE_PLUGIN) in hook_types
    assert ("UserPromptSubmit", SOURCE_PROJECT) in hook_types

    doc_kinds = {(d.kind, d.source) for d in data.context_docs}
    assert ("claude_md", SOURCE_USER) in doc_kinds
    assert ("claude_md", SOURCE_PROJECT) in doc_kinds
    assert ("agents_md", SOURCE_PROJECT) in doc_kinds

    settings_names = {(s.name, s.source) for s in data.settings_files}
    assert ("settings.json", SOURCE_USER) in settings_names
    assert ("settings.local.json", SOURCE_PROJECT) in settings_names


def test_scan_empty_project(tmp_path: Path) -> None:
    project = tmp_path / "empty"
    project.mkdir()
    scanner = ConfigScanner(user_home=tmp_path / "no-home")
    data = scanner.scan(project)
    counts = data.counts()
    assert all(v == 0 for v in counts.values())
