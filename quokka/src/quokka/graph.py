"""Build a renderable graph (nodes + edges) from :class:`HarnessData`.

The four edge types correspond to the relationships the user selected:

* ``provides`` — Plugin → its child Skill / Agent / Command / Hook / MCP.
* ``triggers_on`` — Hook → the tool (or event, if matcher is empty) it
  fires on.
* ``grants_tool`` — Agent → the tools / MCPs listed in its frontmatter.
* ``references`` — context-doc → entities mentioned by name or slash
  command.

Synthetic nodes (``tool:Bash``, ``event:Stop``, etc.) are de-duplicated so
multiple agents/hooks pointing at the same tool share one node.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .references import KnownNames, find_references
from .scanner import (
    Agent,
    ContextDoc,
    HarnessData,
    Hook,
    Mcp,
    Plugin,
)

_MCP_TOOL_RE = re.compile(r"^mcp__([A-Za-z0-9_-]+)__")
_MATCHER_SPLIT_RE = re.compile(r"[,|]")


@dataclass
class _BuildState:
    nodes: dict[str, dict[str, Any]]
    edges: list[dict[str, Any]]
    mcp_by_name: dict[str, Mcp]


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------


def _entity_node(entity: Any, type_label: str) -> dict[str, Any]:
    """Serialise a scanner entity into a node dict."""
    name = getattr(entity, "name", None) or getattr(entity, "kind", "")
    description = getattr(entity, "description", "") or ""
    path = getattr(entity, "path", None)
    meta: dict[str, Any] = {}

    if isinstance(entity, Plugin):
        meta = {
            "version": entity.version,
            "author": entity.author,
            "keywords": entity.keywords,
            "install_path": str(entity.install_path),
        }
        description = entity.description
    elif isinstance(entity, Hook):
        meta = {
            "hook_type": entity.hook_type,
            "matcher": entity.matcher,
            "handler": entity.handler,
            "declared_in": str(entity.declared_in) if entity.declared_in else None,
            "plugin": entity.plugin_name,
        }
        description = f"{entity.hook_type} hook"
        if entity.matcher:
            description += f" on {entity.matcher}"
    elif isinstance(entity, Agent):
        meta = {"model": entity.model, "tools": entity.tools, "plugin": entity.plugin_name}
    elif isinstance(entity, Mcp):
        meta = {
            "command": entity.command,
            "args": entity.args,
            "plugin": entity.plugin_name,
        }
    elif isinstance(entity, ContextDoc):
        meta = {"kind": entity.kind}
        # Show first non-blank line as description.
        for line in (entity.content or "").splitlines():
            line = line.strip().lstrip("#").strip()
            if line:
                description = line[:200]
                break

    return {
        "id": entity.id,
        "type": type_label,
        "name": name,
        "label": name,
        "source": getattr(entity, "source", "user"),
        "path": str(path) if path else None,
        "description": description,
        "meta": meta,
    }


def _ensure_synthetic(
    state: _BuildState,
    *,
    type_label: str,
    name: str,
    description: str,
) -> str:
    """Insert (if absent) a synthetic node such as ``tool:Bash`` and return its id."""
    safe = re.sub(r"[^A-Za-z0-9_.\-]+", "_", name)
    node_id = f"{type_label}:synthetic:{safe}"
    if node_id not in state.nodes:
        state.nodes[node_id] = {
            "id": node_id,
            "type": type_label,
            "name": name,
            "label": name,
            "source": "synthetic",
            "path": None,
            "description": description,
            "meta": {},
        }
    return node_id


def _add_edge(
    state: _BuildState,
    *,
    source: str,
    target: str,
    edge_type: str,
    label: str | None = None,
) -> None:
    if source == target:
        return
    edge_id = f"{source}->{target}:{edge_type}"
    if any(e["id"] == edge_id for e in state.edges):
        return
    state.edges.append(
        {
            "id": edge_id,
            "source": source,
            "target": target,
            "type": edge_type,
            "label": label or edge_type,
        }
    )


# ---------------------------------------------------------------------------
# Edge builders
# ---------------------------------------------------------------------------


def _is_under(child: Path | None, parent: Path) -> bool:
    if child is None:
        return False
    try:
        child.resolve().relative_to(parent.resolve())
    except (ValueError, OSError):
        return False
    return True


def _edges_provides(state: _BuildState, data: HarnessData) -> None:
    plugins_by_path = {p.install_path.resolve(): p for p in data.plugins}
    if not plugins_by_path:
        return

    artifacts: list[tuple[Any, Path | None]] = []
    artifacts.extend((s, s.path) for s in data.skills)
    artifacts.extend((a, a.path) for a in data.agents)
    artifacts.extend((c, c.path) for c in data.commands)
    artifacts.extend((h, h.handler_path) for h in data.hooks)
    artifacts.extend((m, m.config_path) for m in data.mcps)

    for entity, path in artifacts:
        if path is None:
            # Hooks declared in plugin hooks.json have ``declared_in`` populated.
            if isinstance(entity, Hook) and entity.declared_in:
                path = entity.declared_in
            else:
                # Fall back to any plugin name attribute the artifact carries.
                plugin_name = getattr(entity, "plugin_name", None)
                if plugin_name:
                    for plugin in data.plugins:
                        if plugin.name == plugin_name:
                            _add_edge(
                                state,
                                source=plugin.id,
                                target=entity.id,
                                edge_type="provides",
                            )
                            break
                continue
        for plugin_path, plugin in plugins_by_path.items():
            if _is_under(path, plugin_path):
                _add_edge(
                    state,
                    source=plugin.id,
                    target=entity.id,
                    edge_type="provides",
                )
                break


def _resolve_tool_token(
    state: _BuildState, token: str
) -> tuple[str | None, str | None]:
    """Map a hook-matcher / agent-tools token to a node id.

    Returns ``(node_id, label)`` or ``(None, None)`` for empty input.
    """
    token = token.strip()
    if not token:
        return None, None

    m = _MCP_TOOL_RE.match(token)
    if m:
        srv = m.group(1)
        existing = state.mcp_by_name.get(srv)
        if existing is not None:
            return existing.id, srv
        # Materialise a synthetic MCP node for servers we didn't otherwise know.
        node_id = _ensure_synthetic(
            state, type_label="mcp", name=srv, description="MCP server"
        )
        return node_id, srv

    node_id = _ensure_synthetic(
        state, type_label="tool", name=token, description="Tool name"
    )
    return node_id, token


def _edges_triggers_on(state: _BuildState, data: HarnessData) -> None:
    for hook in data.hooks:
        matcher = hook.matcher.strip() if hook.matcher else ""
        if not matcher:
            event_id = _ensure_synthetic(
                state,
                type_label="event",
                name=hook.hook_type,
                description=f"{hook.hook_type} session event",
            )
            _add_edge(
                state,
                source=hook.id,
                target=event_id,
                edge_type="triggers_on",
                label=hook.hook_type,
            )
            continue
        for token in _MATCHER_SPLIT_RE.split(matcher):
            target_id, _ = _resolve_tool_token(state, token)
            if target_id:
                _add_edge(
                    state,
                    source=hook.id,
                    target=target_id,
                    edge_type="triggers_on",
                )


def _edges_grants_tool(state: _BuildState, data: HarnessData) -> None:
    for agent in data.agents:
        for token in agent.tools:
            target_id, _ = _resolve_tool_token(state, token)
            if target_id:
                _add_edge(
                    state,
                    source=agent.id,
                    target=target_id,
                    edge_type="grants_tool",
                )


def _edges_references(state: _BuildState, data: HarnessData) -> None:
    if not data.context_docs:
        return

    known = KnownNames(
        commands=frozenset(c.name for c in data.commands),
        skills=frozenset(s.name for s in data.skills),
        agents=frozenset(a.name for a in data.agents),
        mcps=frozenset(m.name for m in data.mcps),
    )

    targets_by_type: dict[str, dict[str, str]] = {
        "command": {c.name: c.id for c in data.commands},
        "skill": {s.name: s.id for s in data.skills},
        "agent": {a.name: a.id for a in data.agents},
        "mcp": {m.name: m.id for m in data.mcps},
    }

    for doc in data.context_docs:
        for ref in find_references(doc.content or "", known):
            target_id = targets_by_type.get(ref.target_type, {}).get(ref.target_name)
            if target_id is None:
                continue
            _add_edge(
                state,
                source=doc.id,
                target=target_id,
                edge_type="references",
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


_TYPE_LABELS: list[tuple[str, str]] = [
    ("plugins", "plugin"),
    ("skills", "skill"),
    ("agents", "agent"),
    ("commands", "command"),
    ("hooks", "hook"),
    ("mcps", "mcp"),
    ("context_docs", "context_doc"),
    ("settings_files", "settings"),
]


def build_graph(data: HarnessData) -> dict[str, Any]:
    """Return ``{"nodes": [...], "edges": [...], "counts": {...}}``."""
    nodes: dict[str, dict[str, Any]] = {}
    state = _BuildState(
        nodes=nodes,
        edges=[],
        mcp_by_name={m.name: m for m in data.mcps},
    )

    for attr, label in _TYPE_LABELS:
        for entity in getattr(data, attr):
            node = _entity_node(entity, label)
            nodes[node["id"]] = node

    _edges_provides(state, data)
    _edges_triggers_on(state, data)
    _edges_grants_tool(state, data)
    _edges_references(state, data)

    return {
        "nodes": list(nodes.values()),
        "edges": state.edges,
        "counts": data.counts(),
        "project": str(data.project),
        "user_home": str(data.user_home),
    }
