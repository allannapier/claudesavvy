"""Text mining for context-doc references.

Scans the body of a context document (CLAUDE.md / AGENTS.md) for mentions of
known harness entities and returns one ``Ref`` per unique hit.

Two passes:

1. **Slash command tokens** — ``/<name>``.  We anchor on a negative
   lookbehind for ``[A-Za-z0-9_]`` so URL paths like ``/docs/foo`` don't
   match (the leading ``s`` of ``docs/foo`` blocks the match).  We then
   gate hits against the actual ``Command.name`` set so unknown
   slash-prefixed tokens are silently dropped.
2. **Whole-word names** — for each known Skill, Agent, MCP we run
   ``\\b<name>\\b`` on the (code-fence-stripped) text.  Names shorter
   than ``MIN_NAME_LEN`` are skipped to avoid noise on short common
   strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NamedTuple

MIN_NAME_LEN = 4

_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
_SLASH_RE = re.compile(r"(?<![A-Za-z0-9_])/([a-z][a-z0-9_-]{1,40})\b")


class KnownNames(NamedTuple):
    commands: frozenset[str]
    skills: frozenset[str]
    agents: frozenset[str]
    mcps: frozenset[str]


@dataclass(frozen=True)
class Ref:
    """A single reference detected in a context-doc body."""

    target_type: str  # "command" | "skill" | "agent" | "mcp"
    target_name: str


def _strip_code(text: str) -> str:
    text = _FENCE_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(" ", text)
    return text


def find_references(text: str, known: KnownNames) -> list[Ref]:
    """Return de-duplicated references from ``text`` to entities in ``known``."""
    cleaned = _strip_code(text)
    found: set[Ref] = set()

    for match in _SLASH_RE.finditer(cleaned):
        name = match.group(1)
        if name in known.commands:
            found.add(Ref(target_type="command", target_name=name))

    for target_type, names in (
        ("skill", known.skills),
        ("agent", known.agents),
        ("mcp", known.mcps),
    ):
        for name in names:
            if len(name) < MIN_NAME_LEN:
                continue
            pattern = re.compile(rf"\b{re.escape(name)}\b")
            if pattern.search(cleaned):
                found.add(Ref(target_type=target_type, target_name=name))

    return sorted(found, key=lambda r: (r.target_type, r.target_name))
