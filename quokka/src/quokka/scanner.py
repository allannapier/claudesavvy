"""Filesystem scanner for Claude Code harness artifacts.

Discovers every file that influences a Claude Code session at either user
scope (``~/.claude/``) or project scope (``<repo>/.claude/`` and friends),
plus context docs (``CLAUDE.md``, ``AGENTS.md``, ``agents.md``) at the user
home and project root, and any plugin contributions discovered via
``~/.claude/plugins/installed_plugins.json``.

Every read is tolerant: missing files simply yield no entry.  The resulting
:class:`HarnessData` is a plain data container with no Flask / web
dependencies, so it is straightforward to unit-test and to consume from any
renderer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Entity dataclasses
# ---------------------------------------------------------------------------

# `source` values used everywhere
SOURCE_USER = "user"
SOURCE_PROJECT = "project"
SOURCE_PLUGIN = "plugin"


def _stable_id(node_type: str, source: str, name: str) -> str:
    """Build a stable graph-node id of the form ``type:source:name``."""
    safe = re.sub(r"[^A-Za-z0-9_.\-]+", "_", name)
    return f"{node_type}:{source}:{safe}"


@dataclass
class Skill:
    name: str
    description: str
    source: str
    path: Path
    plugin_name: str | None = None
    agents: list[str] = field(default_factory=list)
    content: str = ""

    @property
    def id(self) -> str:
        return _stable_id("skill", self.source, self.name)


@dataclass
class Agent:
    name: str
    description: str
    source: str
    path: Path
    model: str = "inherit"
    tools: list[str] = field(default_factory=list)
    plugin_name: str | None = None
    content: str = ""

    @property
    def id(self) -> str:
        return _stable_id("agent", self.source, self.name)


@dataclass
class Command:
    name: str
    description: str
    source: str
    path: Path
    plugin_name: str | None = None
    content: str = ""

    @property
    def id(self) -> str:
        return _stable_id("command", self.source, self.name)


@dataclass
class Hook:
    name: str
    hook_type: str
    matcher: str
    source: str
    handler: str
    handler_path: Path | None = None
    plugin_name: str | None = None
    declared_in: Path | None = None  # which settings.json / hooks.json declared it

    @property
    def id(self) -> str:
        return _stable_id("hook", self.source, self.name)


@dataclass
class Mcp:
    name: str
    source: str
    command: str = ""
    args: list[str] = field(default_factory=list)
    plugin_name: str | None = None
    config_path: Path | None = None

    @property
    def id(self) -> str:
        return _stable_id("mcp", self.source, self.name)


@dataclass
class Plugin:
    name: str
    version: str
    description: str
    install_path: Path
    enabled: bool = True
    author: str | None = None
    keywords: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return _stable_id("plugin", "plugin", self.name)


@dataclass
class ContextDoc:
    """A free-text doc that influences the harness — CLAUDE.md or AGENTS.md."""

    kind: str  # "claude_md" or "agents_md"
    source: str
    path: Path
    content: str = ""

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def id(self) -> str:
        return _stable_id("context", self.source, self.kind)


@dataclass
class SettingsFile:
    name: str
    source: str
    path: Path
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return _stable_id("settings", self.source, self.name)


@dataclass
class HarnessData:
    project: Path
    user_home: Path
    skills: list[Skill] = field(default_factory=list)
    agents: list[Agent] = field(default_factory=list)
    commands: list[Command] = field(default_factory=list)
    hooks: list[Hook] = field(default_factory=list)
    mcps: list[Mcp] = field(default_factory=list)
    plugins: list[Plugin] = field(default_factory=list)
    context_docs: list[ContextDoc] = field(default_factory=list)
    settings_files: list[SettingsFile] = field(default_factory=list)

    def all_entities(self) -> list[Any]:
        return [
            *self.skills,
            *self.agents,
            *self.commands,
            *self.hooks,
            *self.mcps,
            *self.plugins,
            *self.context_docs,
            *self.settings_files,
        ]

    def counts(self) -> dict[str, int]:
        return {
            "skills": len(self.skills),
            "agents": len(self.agents),
            "commands": len(self.commands),
            "hooks": len(self.hooks),
            "mcps": len(self.mcps),
            "plugins": len(self.plugins),
            "context_docs": len(self.context_docs),
            "settings_files": len(self.settings_files),
        }


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------

_LIST_TOKEN_RE = re.compile(r"[\[\]\"']")


def _split_list(value: str) -> list[str]:
    """Parse a YAML-ish inline list (``[a, b]``) or comma string into clean tokens."""
    cleaned = _LIST_TOKEN_RE.sub("", value)
    return [tok.strip() for tok in cleaned.split(",") if tok.strip()]


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse a tiny subset of YAML frontmatter at the top of a markdown file.

    Returns ``(frontmatter_dict, body)``.  Only a single-line ``key: value``
    form per field is recognised, which is sufficient for the fields used in
    SKILL.md, agent definitions, and slash command headers.
    """
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    frontmatter: dict[str, Any] = {}
    for raw_line in parts[1].splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        frontmatter[key.strip()] = value.strip()

    return frontmatter, parts[2].lstrip("\n")


def _first_meaningful_line(text: str, limit: int = 200) -> str:
    for raw in text.splitlines():
        line = raw.strip().lstrip("#").strip()
        if line and not line.startswith("---"):
            return line[:limit]
    return ""


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


class ConfigScanner:
    """Discover every harness artifact for a given project + user home."""

    def __init__(self, user_home: Path | None = None) -> None:
        self.user_home = user_home if user_home is not None else Path.home()
        self.user_claude = self.user_home / ".claude"

    # -- public entry point ------------------------------------------------

    def scan(self, project: Path) -> HarnessData:
        project = project.resolve()
        data = HarnessData(project=project, user_home=self.user_home)

        plugin_dirs = self._scan_plugins(data)

        self._scan_settings(data, project)
        self._scan_context_docs(data, project, plugin_dirs)
        self._scan_skills(data, project, plugin_dirs)
        self._scan_agents(data, project, plugin_dirs)
        self._scan_commands(data, project, plugin_dirs)
        self._scan_hooks(data, project, plugin_dirs)
        self._scan_mcps(data, project, plugin_dirs)

        return data

    # -- plugins -----------------------------------------------------------

    def _scan_plugins(self, data: HarnessData) -> dict[str, Path]:
        """Return a ``plugin_name -> install_path`` map. Populates ``data.plugins``."""
        plugin_dirs: dict[str, Path] = {}
        manifest = self.user_claude / "plugins" / "installed_plugins.json"
        if not manifest.exists():
            return plugin_dirs

        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return plugin_dirs

        for name, versions in (payload.get("plugins") or {}).items():
            if not versions:
                continue
            latest = versions[0]
            install_path = Path(latest.get("installPath", "")).expanduser()
            if not install_path.exists():
                continue

            description = ""
            author: str | None = None
            keywords: list[str] = []
            plugin_json = install_path / ".claude-plugin" / "plugin.json"
            if plugin_json.exists():
                try:
                    meta = json.loads(plugin_json.read_text(encoding="utf-8"))
                    description = meta.get("description", "") or ""
                    raw_author = meta.get("author")
                    if isinstance(raw_author, dict):
                        author = raw_author.get("name")
                    elif isinstance(raw_author, str):
                        author = raw_author
                    keywords = list(meta.get("keywords") or [])
                except (OSError, json.JSONDecodeError):
                    pass

            data.plugins.append(
                Plugin(
                    name=name,
                    version=latest.get("version", "unknown"),
                    description=description,
                    install_path=install_path,
                    author=author,
                    keywords=keywords,
                )
            )
            plugin_dirs[name] = install_path

        return plugin_dirs

    # -- settings ----------------------------------------------------------

    def _scan_settings(self, data: HarnessData, project: Path) -> None:
        candidates = [
            (self.user_claude / "settings.json", SOURCE_USER, "settings.json"),
            (project / ".claude" / "settings.json", SOURCE_PROJECT, "settings.json"),
            (
                project / ".claude" / "settings.local.json",
                SOURCE_PROJECT,
                "settings.local.json",
            ),
        ]
        for path, source, name in candidates:
            if not path.exists():
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raw = {}
            data.settings_files.append(
                SettingsFile(name=name, source=source, path=path, raw=raw)
            )

    # -- context docs ------------------------------------------------------

    _CONTEXT_KIND_BY_FILENAME = {
        "CLAUDE.md": "claude_md",
        "AGENTS.md": "agents_md",
        "agents.md": "agents_md",
    }

    def _read_context_doc(
        self, path: Path, source: str
    ) -> ContextDoc | None:
        if not path.exists() or not path.is_file():
            return None
        kind = self._CONTEXT_KIND_BY_FILENAME.get(path.name)
        if kind is None:
            return None
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            return None
        return ContextDoc(kind=kind, source=source, path=path, content=content)

    def _scan_context_docs(
        self,
        data: HarnessData,
        project: Path,
        plugin_dirs: dict[str, Path],
    ) -> None:
        for fname in self._CONTEXT_KIND_BY_FILENAME:
            doc = self._read_context_doc(self.user_claude / fname, SOURCE_USER)
            if doc:
                data.context_docs.append(doc)
            doc = self._read_context_doc(project / fname, SOURCE_PROJECT)
            if doc:
                data.context_docs.append(doc)
            for plugin_path in plugin_dirs.values():
                doc = self._read_context_doc(plugin_path / fname, SOURCE_PLUGIN)
                if doc:
                    data.context_docs.append(doc)

    # -- skills ------------------------------------------------------------

    def _parse_skill(
        self, skill_dir: Path, source: str, plugin_name: str | None = None
    ) -> Skill | None:
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            skill_md = skill_dir / "SKILLS.md"
        if not skill_md.exists():
            return None
        try:
            content = skill_md.read_text(encoding="utf-8")
        except OSError:
            return None
        fm, body = parse_frontmatter(content)
        agents = _split_list(fm.get("agents", "")) if fm.get("agents") else []
        description = fm.get("description") or _first_meaningful_line(body)
        return Skill(
            name=skill_dir.name,
            description=description,
            source=source,
            path=skill_md,
            plugin_name=plugin_name,
            agents=agents,
            content=content,
        )

    def _scan_skills(
        self,
        data: HarnessData,
        project: Path,
        plugin_dirs: dict[str, Path],
    ) -> None:
        roots: list[tuple[Path, str, str | None]] = [
            (self.user_claude / "skills", SOURCE_USER, None),
            (project / ".claude" / "skills", SOURCE_PROJECT, None),
        ]
        for plugin_name, plugin_path in plugin_dirs.items():
            roots.append((plugin_path / "skills", SOURCE_PLUGIN, plugin_name))

        for root, source, plugin_name in roots:
            if not root.exists() or not root.is_dir():
                continue
            for child in sorted(root.iterdir()):
                if not child.is_dir():
                    continue
                skill = self._parse_skill(child, source, plugin_name)
                if skill:
                    data.skills.append(skill)

    # -- agents (sub-agent definitions) ------------------------------------

    def _parse_agent(
        self, path: Path, source: str, plugin_name: str | None = None
    ) -> Agent | None:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            return None
        fm, body = parse_frontmatter(content)
        tools = _split_list(fm.get("tools", "")) if fm.get("tools") else []
        description = fm.get("description") or _first_meaningful_line(body)
        return Agent(
            name=path.stem,
            description=description,
            source=source,
            path=path,
            model=fm.get("model", "inherit"),
            tools=tools,
            plugin_name=plugin_name,
            content=content,
        )

    def _scan_agents(
        self,
        data: HarnessData,
        project: Path,
        plugin_dirs: dict[str, Path],
    ) -> None:
        roots: list[tuple[Path, str, str | None]] = [
            (self.user_claude / "agents", SOURCE_USER, None),
            (project / ".claude" / "agents", SOURCE_PROJECT, None),
        ]
        for plugin_name, plugin_path in plugin_dirs.items():
            roots.append((plugin_path / "agents", SOURCE_PLUGIN, plugin_name))

        for root, source, plugin_name in roots:
            if not root.exists() or not root.is_dir():
                continue
            for path in sorted(root.glob("*.md")):
                agent = self._parse_agent(path, source, plugin_name)
                if agent:
                    data.agents.append(agent)

    # -- commands ----------------------------------------------------------

    def _parse_command(
        self, path: Path, source: str, plugin_name: str | None = None
    ) -> Command | None:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            return None
        fm, body = parse_frontmatter(content)
        description = fm.get("description") or _first_meaningful_line(body)
        return Command(
            name=path.stem,
            description=description,
            source=source,
            path=path,
            plugin_name=plugin_name,
            content=content,
        )

    def _scan_commands(
        self,
        data: HarnessData,
        project: Path,
        plugin_dirs: dict[str, Path],
    ) -> None:
        roots: list[tuple[Path, str, str | None]] = [
            (self.user_claude / "commands", SOURCE_USER, None),
            (project / ".claude" / "commands", SOURCE_PROJECT, None),
        ]
        for plugin_name, plugin_path in plugin_dirs.items():
            roots.append((plugin_path / "commands", SOURCE_PLUGIN, plugin_name))

        for root, source, plugin_name in roots:
            if not root.exists() or not root.is_dir():
                continue
            for path in sorted(root.glob("*.md")):
                cmd = self._parse_command(path, source, plugin_name)
                if cmd:
                    data.commands.append(cmd)

    # -- hooks -------------------------------------------------------------

    def _hooks_from_settings_block(
        self,
        block: dict[str, Any],
        source: str,
        declared_in: Path,
        plugin_name: str | None = None,
    ) -> list[Hook]:
        """Materialise Hook entries from a ``hooks: { EventName: [ ... ] }`` block."""
        out: list[Hook] = []
        if not isinstance(block, dict):
            return out
        for hook_type, configs in block.items():
            if not isinstance(configs, list):
                continue
            for idx, cfg in enumerate(configs):
                if not isinstance(cfg, dict):
                    continue
                matcher = str(cfg.get("matcher", "") or "")
                inner = cfg.get("hooks") or []
                if not isinstance(inner, list) or not inner:
                    continue
                for j, entry in enumerate(inner):
                    if not isinstance(entry, dict):
                        continue
                    handler = str(entry.get("command") or entry.get("type") or "")
                    handler_path: Path | None = None
                    if "command" in entry and isinstance(entry["command"], str):
                        cmd_str = entry["command"]
                        if "${CLAUDE_PLUGIN_ROOT}" in cmd_str and plugin_name:
                            # plugin-relative path; not always parseable, best effort
                            handler_path = None
                    suffix_parts = [hook_type]
                    if matcher:
                        suffix_parts.append(matcher)
                    suffix_parts.append(str(idx))
                    if len(inner) > 1:
                        suffix_parts.append(str(j))
                    suffix = "_".join(re.sub(r"[^A-Za-z0-9]+", "-", p) for p in suffix_parts)
                    name = f"{plugin_name}_{suffix}" if plugin_name else suffix
                    out.append(
                        Hook(
                            name=name,
                            hook_type=hook_type,
                            matcher=matcher,
                            source=source,
                            handler=handler,
                            handler_path=handler_path,
                            plugin_name=plugin_name,
                            declared_in=declared_in,
                        )
                    )
        return out

    def _scan_hooks(
        self,
        data: HarnessData,
        project: Path,
        plugin_dirs: dict[str, Path],
    ) -> None:
        # Hooks declared in user/project settings.json
        for sf in data.settings_files:
            block = sf.raw.get("hooks") if isinstance(sf.raw, dict) else None
            data.hooks.extend(
                self._hooks_from_settings_block(block, sf.source, sf.path)
                if block
                else []
            )

        # Hooks declared by plugins (hooks/hooks.json)
        for plugin_name, plugin_path in plugin_dirs.items():
            hooks_json = plugin_path / "hooks" / "hooks.json"
            if not hooks_json.exists():
                continue
            try:
                payload = json.loads(hooks_json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            block = payload.get("hooks") if isinstance(payload, dict) else None
            if block:
                data.hooks.extend(
                    self._hooks_from_settings_block(
                        block,
                        SOURCE_PLUGIN,
                        hooks_json,
                        plugin_name=plugin_name,
                    )
                )

    # -- MCPs --------------------------------------------------------------

    _MCP_PERM_RE = re.compile(r"^mcp__([A-Za-z0-9_-]+)__")

    def _mcp_names_from_settings(self, raw: dict[str, Any]) -> set[str]:
        out: set[str] = set()
        perms = raw.get("permissions") if isinstance(raw, dict) else None
        if not isinstance(perms, dict):
            return out
        for kind in ("allow", "ask", "deny"):
            for entry in perms.get(kind, []) or []:
                if isinstance(entry, str):
                    m = self._MCP_PERM_RE.match(entry)
                    if m:
                        out.add(m.group(1))
        return out

    def _scan_mcps(
        self,
        data: HarnessData,
        project: Path,
        plugin_dirs: dict[str, Path],
    ) -> None:
        seen: dict[str, Mcp] = {}

        # Settings-derived MCP names
        for sf in data.settings_files:
            for name in self._mcp_names_from_settings(sf.raw):
                if name in seen:
                    continue
                seen[name] = Mcp(
                    name=name,
                    source=sf.source,
                    config_path=sf.path,
                )

        # Plugin .mcp.json files
        for plugin_name, plugin_path in plugin_dirs.items():
            mcp_json = plugin_path / ".mcp.json"
            if not mcp_json.exists():
                continue
            try:
                payload = json.loads(mcp_json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            servers = payload.get("mcpServers") if isinstance(payload, dict) else payload
            if not isinstance(servers, dict):
                continue
            for name, cfg in servers.items():
                if not isinstance(cfg, dict):
                    continue
                if name in seen:
                    # Plugin definition fills in missing command/args metadata.
                    existing = seen[name]
                    if not existing.command:
                        existing.command = str(cfg.get("command", ""))
                        existing.args = list(cfg.get("args") or [])
                        existing.plugin_name = plugin_name
                        existing.config_path = mcp_json
                    continue
                seen[name] = Mcp(
                    name=name,
                    source=SOURCE_PLUGIN,
                    command=str(cfg.get("command", "")),
                    args=list(cfg.get("args") or []),
                    plugin_name=plugin_name,
                    config_path=mcp_json,
                )

        data.mcps.extend(seen.values())
