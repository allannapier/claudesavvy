"""Parser for Claude Code skills."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SkillInfo:
    """Information about an installed skill."""

    name: str
    path: Path
    description: str = ""
    has_skills_md: bool = False

    @classmethod
    def from_directory(cls, skill_dir: Path) -> Optional["SkillInfo"]:
        """
        Create SkillInfo from a skill directory.

        Args:
            skill_dir: Path to skill directory

        Returns:
            SkillInfo instance or None if invalid
        """
        if not skill_dir.is_dir():
            return None

        name = skill_dir.name

        # Check for SKILL.md (singular - used in plugins)
        skill_md = skill_dir / "SKILL.md"
        # Also check for SKILLS.md (plural - legacy)
        skills_md = skill_dir / "SKILLS.md"

        has_skill_file = skill_md.exists() or skills_md.exists()
        skill_file = skill_md if skill_md.exists() else skills_md

        description = ""
        if has_skill_file:
            try:
                with open(skill_file, 'r', encoding='utf-8') as f:
                    # Read first line as description
                    first_line = f.readline().strip()
                    # Remove markdown header markers
                    description = first_line.lstrip('#').strip()
            except OSError:
                # Description is optional; use empty string as default
                pass

        return cls(
            name=name,
            path=skill_dir,
            description=description,
            has_skills_md=has_skill_file
        )


class SkillsParser:
    """Parser for Claude Code skills."""

    def __init__(self, skills_dir: Path):
        """
        Initialize skills parser.

        Args:
            skills_dir: Path to ~/.claude/skills directory
        """
        self.skills_dir = skills_dir
        self.claude_dir = skills_dir.parent
        self.plugins_dir = self.claude_dir / "plugins"

    def get_installed_skills(self) -> list[SkillInfo]:
        """
        Get list of installed skills from both global skills directory
        and installed plugins.

        Returns:
            List of SkillInfo instances
        """
        skills = []

        # Check global skills directory
        if self.skills_dir.exists():
            for skill_dir in self.skills_dir.iterdir():
                if skill_dir.is_dir() and not skill_dir.name.startswith('.'):
                    skill_info = SkillInfo.from_directory(skill_dir)
                    if skill_info:
                        skills.append(skill_info)

        # Check installed plugins for skills
        skills.extend(self._get_plugin_skills())

        return sorted(skills, key=lambda s: s.name)

    def _get_plugin_skills(self) -> list[SkillInfo]:
        """
        Get skills from installed plugins.

        Returns:
            List of SkillInfo instances from plugins
        """
        skills = []
        installed_plugins_file = self.plugins_dir / "installed_plugins.json"

        if not installed_plugins_file.exists():
            return skills

        try:
            with open(installed_plugins_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            plugins = data.get('plugins', {})
            for plugin_name, plugin_info in plugins.items():
                install_path = Path(plugin_info.get('installPath', ''))
                if install_path.exists():
                    skills_dir = install_path / 'skills'
                    if skills_dir.exists():
                        for skill_dir in skills_dir.iterdir():
                            if skill_dir.is_dir() and not skill_dir.name.startswith('.'):
                                skill_info = SkillInfo.from_directory(skill_dir)
                                if skill_info:
                                    skills.append(skill_info)
        except (OSError, json.JSONDecodeError):
            # Failed to read plugin data; skip this plugin
            pass

        return skills

    def get_skill_count(self) -> int:
        """
        Get total number of installed skills.

        Returns:
            Number of skills
        """
        return len(self.get_installed_skills())


class ConfigurationParser:
    """Parser for Claude Code configuration files."""

    def __init__(self, claude_dir: Path):
        """
        Initialize configuration parser.

        Args:
            claude_dir: Path to ~/.claude directory
        """
        self.claude_dir = claude_dir
        self.settings_file = claude_dir / "settings.json"

    def get_global_settings(self) -> dict:
        """
        Get global Claude Code settings.

        Returns:
            Dict with settings data
        """
        if not self.settings_file.exists():
            return {}

        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            # Failed to read settings file; return empty dict
            return {}

    def get_enabled_plugins(self) -> dict[str, bool]:
        """
        Get enabled plugins from settings.

        Returns:
            Dict mapping plugin names to enabled status
        """
        settings = self.get_global_settings()
        return settings.get('enabledPlugins', {})

    def is_always_thinking_enabled(self) -> bool:
        """
        Check if always thinking mode is enabled.

        Returns:
            True if enabled
        """
        settings = self.get_global_settings()
        return settings.get('alwaysThinkingEnabled', False)

    def get_status_line_config(self) -> dict:
        """
        Get status line configuration.

        Returns:
            Dict with status line config
        """
        settings = self.get_global_settings()
        return settings.get('statusLine', {})

    @staticmethod
    def find_project_configs(search_path: Path) -> list[Path]:
        """
        Find project-level .claude directories.

        Args:
            search_path: Path to search from (e.g., ~/code)

        Returns:
            List of paths to .claude directories
        """
        configs = []

        if not search_path.exists():
            return configs

        try:
            # Search for .claude directories (max depth 3)
            for item in search_path.rglob('.claude'):
                if item.is_dir():
                    # Check depth
                    depth = len(item.relative_to(search_path).parts)
                    if depth <= 3:
                        configs.append(item)
        except (OSError, PermissionError):
            # Skip directories we cannot access
            pass

        return sorted(configs)

    @staticmethod
    def get_project_settings(project_claude_dir: Path) -> dict:
        """
        Get settings from a project-level .claude directory.

        Args:
            project_claude_dir: Path to project .claude directory

        Returns:
            Dict with project settings
        """
        settings_file = project_claude_dir / "settings.local.json"

        if not settings_file.exists():
            return {}

        try:
            with open(settings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            # Failed to read project settings; return empty dict
            return {}
