"""Graph builder tests."""

from __future__ import annotations

from quokka.graph import build_graph
from quokka.references import KnownNames, find_references
from quokka.scanner import ConfigScanner


def _edge_pairs(graph: dict, edge_type: str) -> set[tuple[str, str]]:
    return {(e["source"], e["target"]) for e in graph["edges"] if e["type"] == edge_type}


def test_build_graph_emits_all_edge_types(synthetic_harness: dict) -> None:
    scanner = ConfigScanner(user_home=synthetic_harness["user_home"])
    data = scanner.scan(synthetic_harness["project"])
    graph = build_graph(data)

    node_ids = {n["id"] for n in graph["nodes"]}
    plugin_id = next(n["id"] for n in graph["nodes"] if n["type"] == "plugin")
    skill_beta = next(n["id"] for n in graph["nodes"] if n["type"] == "skill" and n["name"] == "beta")
    skill_alpha = next(n["id"] for n in graph["nodes"] if n["type"] == "skill" and n["name"] == "alpha")
    command_deploy = next(n["id"] for n in graph["nodes"] if n["type"] == "command" and n["name"] == "deploy")
    command_ship = next(n["id"] for n in graph["nodes"] if n["type"] == "command" and n["name"] == "ship")
    agent_reviewer = next(n["id"] for n in graph["nodes"] if n["type"] == "agent" and n["name"] == "reviewer")
    mcp_github = next(n["id"] for n in graph["nodes"] if n["type"] == "mcp" and n["name"] == "github")
    user_claude_md = next(
        n["id"]
        for n in graph["nodes"]
        if n["type"] == "context_doc" and n["source"] == "user"
    )
    project_agents_md = next(
        n["id"]
        for n in graph["nodes"]
        if n["type"] == "context_doc" and n["source"] == "project" and n["meta"]["kind"] == "agents_md"
    )

    # provides edges
    provides = _edge_pairs(graph, "provides")
    assert (plugin_id, skill_beta) in provides
    assert (plugin_id, command_deploy) in provides
    assert (plugin_id, mcp_github) in provides

    # triggers_on edges (Hook -> tool/event)
    triggers = {(graph_lookup_type(graph, e["source"]), graph_lookup_label(graph, e["target"]))
                for e in graph["edges"] if e["type"] == "triggers_on"}
    assert ("hook", "Bash") in triggers      # PreToolUse + matcher Bash
    assert ("hook", "Edit") in triggers      # plugin PostToolUse + matcher Edit
    assert ("hook", "Stop") in triggers      # Stop hook with empty matcher → event:Stop
    assert ("hook", "UserPromptSubmit") in triggers

    # grants_tool edges (Agent -> Bash/MCP)
    grants = {(graph_lookup_type(graph, e["source"]), graph_lookup_label(graph, e["target"]))
              for e in graph["edges"] if e["type"] == "grants_tool"}
    assert ("agent", "Bash") in grants
    # Reviewer grants github MCP — should map to existing mcp node, not synthetic.
    assert (agent_reviewer, mcp_github) in {(e["source"], e["target"]) for e in graph["edges"] if e["type"] == "grants_tool"}

    # references edges (CLAUDE.md / AGENTS.md → known entities)
    refs = _edge_pairs(graph, "references")
    # User CLAUDE.md mentions /ship and reviewer
    assert (user_claude_md, command_ship) in refs
    assert (user_claude_md, agent_reviewer) in refs
    # Project AGENTS.md mentions reviewer
    assert (project_agents_md, agent_reviewer) in refs
    # Project CLAUDE.md mentions /ship and the alpha skill name
    project_claude_md = next(
        n["id"]
        for n in graph["nodes"]
        if n["type"] == "context_doc" and n["source"] == "project" and n["meta"]["kind"] == "claude_md"
    )
    assert (project_claude_md, command_ship) in refs
    assert (project_claude_md, skill_alpha) in refs

    # Negative case: no edge created for /docs/foo URL fragment.
    docs_fragments = [e for e in graph["edges"] if e["type"] == "references" and "docs" in e["target"]]
    assert docs_fragments == []

    # Sanity: every edge endpoint is a real node.
    for e in graph["edges"]:
        assert e["source"] in node_ids
        assert e["target"] in node_ids


def graph_lookup_type(graph: dict, node_id: str) -> str:
    for n in graph["nodes"]:
        if n["id"] == node_id:
            return n["type"]
    return ""


def graph_lookup_label(graph: dict, node_id: str) -> str:
    for n in graph["nodes"]:
        if n["id"] == node_id:
            return n["label"]
    return ""


def test_references_filters_unknown_slash_tokens() -> None:
    known = KnownNames(
        commands=frozenset({"ship"}),
        skills=frozenset({"alpha"}),
        agents=frozenset(),
        mcps=frozenset(),
    )
    text = "Use /ship to release. Avoid /docs/foo URLs. Talk about alpha."
    refs = find_references(text, known)
    targets = {(r.target_type, r.target_name) for r in refs}
    assert ("command", "ship") in targets
    assert ("skill", "alpha") in targets
    # /docs and /foo are not registered commands; should be filtered.
    assert all(r.target_name != "docs" for r in refs)
    assert all(r.target_name != "foo" for r in refs)


def test_references_strips_fenced_code_blocks() -> None:
    known = KnownNames(
        commands=frozenset({"ship"}),
        skills=frozenset(),
        agents=frozenset(),
        mcps=frozenset(),
    )
    text = "Outside /ship.\n```\n/ship inside\n```\n"
    refs = find_references(text, known)
    # Still found because there's an outside hit.
    assert any(r.target_name == "ship" for r in refs)
    # And dedup'd to a single entry.
    assert sum(1 for r in refs if r.target_name == "ship") == 1
