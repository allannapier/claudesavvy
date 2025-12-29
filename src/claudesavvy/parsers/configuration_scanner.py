"""Scanner for Claude Code configurations across repositories."""

import json
from pathlib import Path
from typing import List, Optional
from .skills import ConfigurationParser
from ..models import (
    ConfigSource,
    RepositoryConfig,
    Skill,
    MCP,
    Command,
    Plugin,
    Hook,
    Agent,
    InheritanceLevel,
    InheritanceChain,
    ConfigurationFeatures,
)


class ConfigurationScanner:
    """Scans for and parses Claude Code configurations across repositories."""

    def __init__(self, config_parser: Optional[ConfigurationParser] = None):
        """
        Initialize configuration scanner.

        Args:
            config_parser: Optional ConfigurationParser instance
        """
        self.config_parser = config_parser

    def scan_for_repositories(
        self, base_path: Optional[Path] = None, max_depth: int = 3
    ) -> List[RepositoryConfig]:
        """
        Scan for .claude/ directories in the filesystem.

        Args:
            base_path: Base path to scan from (defaults to user home directory)
            max_depth: Maximum directory depth to search

        Returns:
            List of RepositoryConfig objects
        """
        if base_path is None:
            base_path = Path.home()

        repositories = []

        # Always include user's ~/.claude/ if it exists
        user_claude_dir = Path.home() / ".claude"
        if user_claude_dir.exists() and user_claude_dir.is_dir():
            repositories.append(
                RepositoryConfig(
                    path=Path.home(),
                    name="User Configuration",
                    claude_dir=user_claude_dir
                )
            )

        # Common code directories to scan
        common_dirs = [
            base_path / "code",
            base_path / "projects",
            base_path / "repos",
            base_path / "workspace",
            base_path / "dev",
            base_path / "Development",
        ]

        # Scan common directories
        for common_dir in common_dirs:
            if common_dir.exists() and common_dir.is_dir():
                try:
                    repos = self._scan_directory(common_dir, max_depth, current_depth=0)
                    repositories.extend(repos)
                except PermissionError:
                    # Skip directories we can't access
                    pass

        # Also scan base_path root (but not deep)
        try:
            repos = self._scan_directory(base_path, max_depth=1, current_depth=0)
            repositories.extend(repos)
        except PermissionError:
            # Skip directories we can't access
            pass

        # Deduplicate by claude_dir path
        seen_paths = {repo.claude_dir for repo in repositories[:1]}  # Keep user config
        unique_repos = repositories[:1] if repositories else []

        for repo in repositories[1:]:
            if repo.claude_dir not in seen_paths:
                seen_paths.add(repo.claude_dir)
                unique_repos.append(repo)

        return unique_repos

    def _scan_directory(
        self, directory: Path, max_depth: int, current_depth: int
    ) -> List[RepositoryConfig]:
        """Recursively scan a directory for .claude/ folders."""
        repositories = []

        if current_depth > max_depth:
            return repositories

        try:
            for item in directory.iterdir():
                if not item.is_dir():
                    continue

                # Skip hidden directories (except .claude)
                if item.name.startswith('.') and item.name != '.claude':
                    continue

                # Skip common non-code directories
                skip_dirs = {
                    'node_modules', 'venv', '.venv', 'env', '.env',
                    '__pycache__', '.git', 'dist', 'build', 'target',
                    'Library', 'Applications', 'Pictures', 'Music',
                    'Movies', 'Downloads', 'Documents', 'Desktop'
                }
                if item.name in skip_dirs:
                    continue

                # Check if this directory has a .claude/ subdirectory
                claude_dir = item / ".claude"
                if claude_dir.exists() and claude_dir.is_dir():
                    # This is a project with Claude config
                    repositories.append(
                        RepositoryConfig(
                            path=item,
                            name=item.name,
                            claude_dir=claude_dir
                        )
                    )

                # Recurse into subdirectory
                if current_depth < max_depth:
                    sub_repos = self._scan_directory(item, max_depth, current_depth + 1)
                    repositories.extend(sub_repos)

        except PermissionError:
            pass  # Skip directories we can't access

        return repositories

    def get_all_features(self, repo_path: Path) -> ConfigurationFeatures:
        """
        Get all configuration features for a repository.

        Args:
            repo_path: Path to the repository

        Returns:
            ConfigurationFeatures object with all features
        """
        claude_dir = repo_path / ".claude" if repo_path.name != ".claude" else repo_path.parent / ".claude"
        if not claude_dir.exists():
            return ConfigurationFeatures()

        features = ConfigurationFeatures()

        # Get user-level features (from ~/.claude/)
        user_claude_dir = Path.home() / ".claude"

        # Parse skills
        features.skills = self._parse_skills(claude_dir, user_claude_dir)

        # Parse MCPs
        features.mcps = self._parse_mcps(claude_dir, user_claude_dir)

        # Parse commands
        features.commands = self._parse_commands(claude_dir, user_claude_dir)

        # Parse plugins
        features.plugins = self._parse_plugins(user_claude_dir)

        # Parse hooks
        features.hooks = self._parse_hooks(user_claude_dir)

        # Parse agents
        features.agents = self._parse_agents(claude_dir, user_claude_dir)

        return features

    def _parse_skills(self, project_claude_dir: Path, user_claude_dir: Path) -> List[Skill]:
        """Parse skills from user and project directories."""
        skills = []

        # User skills from ~/.claude/skills/
        user_skills_dir = user_claude_dir / "skills"
        if user_skills_dir.exists():
            for skill_dir in user_skills_dir.iterdir():
                if skill_dir.is_dir():
                    skill = self._parse_skill_directory(skill_dir, ConfigSource.USER)
                    if skill:
                        skills.append(skill)

        # Plugin skills from installed plugins
        installed_plugins_file = user_claude_dir / "plugins" / "installed_plugins.json"
        if installed_plugins_file.exists():
            try:
                with open(installed_plugins_file, 'r', encoding='utf-8') as f:
                    plugins_data = json.load(f)
                    for plugin_name, versions in plugins_data.get('plugins', {}).items():
                        if versions:
                            latest_version = versions[0]  # First is usually latest
                            install_path = Path(latest_version.get('installPath', ''))
                            if install_path.exists():
                                plugin_skills_dir = install_path / "skills"
                                if plugin_skills_dir.exists():
                                    for skill_dir in plugin_skills_dir.iterdir():
                                        if skill_dir.is_dir():
                                            skill = self._parse_skill_directory(
                                                skill_dir, ConfigSource.PLUGIN
                                            )
                                            if skill:
                                                skills.append(skill)
            except (OSError, json.JSONDecodeError):
                # Failed to read plugin skills; skip
                pass

        return skills

    def _parse_skill_directory(self, skill_dir: Path, source: ConfigSource) -> Optional[Skill]:
        """Parse a single skill directory."""
        skill_md = skill_dir / "SKILL.md"
        skills_md = skill_dir / "SKILLS.md"

        skill_file = skill_md if skill_md.exists() else (skills_md if skills_md.exists() else None)
        if not skill_file:
            return None

        try:
            with open(skill_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Try to parse YAML frontmatter
            description = ""
            version = None
            agents = []

            if content.startswith('---'):
                # Has YAML frontmatter
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    frontmatter = parts[1]
                    # Simple YAML parsing for key fields
                    for line in frontmatter.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip()
                            value = value.strip()
                            if key == 'description':
                                description = value
                            elif key == 'version':
                                version = value
                            elif key == 'agents':
                                # Parse list
                                agents = [a.strip().strip('[]') for a in value.split(',')]

            if not description:
                # Fallback to first line
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('---'):
                        description = line.lstrip('#').strip()
                        break

            return Skill(
                name=skill_dir.name,
                description=description,
                source=source,
                path=skill_file,
                content=content,
                version=version,
                agents=agents
            )
        except OSError:
            return None

    def _parse_mcps(self, project_claude_dir: Path, user_claude_dir: Path) -> List[MCP]:
        """Parse MCP server configurations from plugins and settings files."""
        mcps = []
        mcp_servers = set()  # Track unique MCP server names

        # Parse MCPs from user settings.json
        user_settings_file = user_claude_dir / "settings.json"
        if user_settings_file.exists():
            try:
                with open(user_settings_file, 'r', encoding='utf-8') as f:
                    settings_data = json.load(f)
                    mcp_servers.update(self._extract_mcp_servers_from_permissions(settings_data))
            except (OSError, json.JSONDecodeError):
                # Failed to read user settings; skip
                pass

        # Parse MCPs from project settings.local.json
        project_settings_file = project_claude_dir / "settings.local.json"
        if project_settings_file.exists():
            try:
                with open(project_settings_file, 'r', encoding='utf-8') as f:
                    settings_data = json.load(f)
                    mcp_servers.update(self._extract_mcp_servers_from_permissions(settings_data))
            except (OSError, json.JSONDecodeError):
                # Failed to read project settings; skip
                pass

        # Create MCP objects for discovered servers from settings
        # Determine source based on where they were found
        for server_name in sorted(mcp_servers):
            # Check if it's in project settings
            source = ConfigSource.PROJECT
            if project_settings_file.exists():
                try:
                    with open(project_settings_file, 'r', encoding='utf-8') as f:
                        project_data = json.load(f)
                        project_mcps = self._extract_mcp_servers_from_permissions(project_data)
                        if server_name not in project_mcps:
                            source = ConfigSource.USER
                except (OSError, json.JSONDecodeError):
                    source = ConfigSource.USER
            else:
                source = ConfigSource.USER

            mcps.append(MCP(
                name=server_name,
                command='',  # Not available from permissions list
                args=[],
                source=source,
                plugin_path=None
            ))

        # Also parse MCPs from plugin .mcp.json files
        installed_plugins_file = user_claude_dir / "plugins" / "installed_plugins.json"
        if installed_plugins_file.exists():
            try:
                with open(installed_plugins_file, 'r', encoding='utf-8') as f:
                    plugins_data = json.load(f)

                for plugin_name, versions in plugins_data.get('plugins', {}).items():
                    if versions:
                        latest_version = versions[0]
                        install_path = Path(latest_version.get('installPath', ''))
                        if install_path.exists():
                            mcp_config_file = install_path / ".mcp.json"
                            if mcp_config_file.exists():
                                with open(mcp_config_file, 'r', encoding='utf-8') as mcp_f:
                                    mcp_config = json.load(mcp_f)
                                    for server_name, config in mcp_config.items():
                                        # Only add if not already found in settings
                                        if server_name not in mcp_servers:
                                            mcps.append(MCP(
                                                name=server_name,
                                                command=config.get('command', ''),
                                                args=config.get('args', []),
                                                source=ConfigSource.PLUGIN,
                                                plugin_path=install_path
                                            ))
            except (OSError, json.JSONDecodeError):
                # Failed to read plugin MCP configs; skip
                pass

        return mcps

    def _extract_mcp_servers_from_permissions(self, settings_data: dict) -> set:
        """Extract MCP server names from permissions list."""
        mcp_servers = set()
        permissions = settings_data.get('permissions', {})

        for perm_type in ['allow', 'deny', 'ask']:
            for permission in permissions.get(perm_type, []):
                if isinstance(permission, str) and permission.startswith('mcp__'):
                    # Format: mcp__<server_name>__<tool_name>
                    parts = permission.split('__')
                    if len(parts) >= 3:
                        server_name = parts[1]
                        mcp_servers.add(server_name)

        return mcp_servers

    def _parse_commands(self, project_claude_dir: Path, user_claude_dir: Path) -> List[Command]:
        """Parse slash commands from user and project directories."""
        commands = []

        # User commands from ~/.claude/commands/
        user_commands_dir = user_claude_dir / "commands"
        if user_commands_dir.exists():
            for cmd_file in user_commands_dir.glob("*.md"):
                command = self._parse_command_file(cmd_file, ConfigSource.USER)
                if command:
                    commands.append(command)

        # Plugin commands
        installed_plugins_file = user_claude_dir / "plugins" / "installed_plugins.json"
        if installed_plugins_file.exists():
            try:
                with open(installed_plugins_file, 'r', encoding='utf-8') as f:
                    plugins_data = json.load(f)
                    for plugin_name, versions in plugins_data.get('plugins', {}).items():
                        if versions:
                            latest_version = versions[0]
                            install_path = Path(latest_version.get('installPath', ''))
                            if install_path.exists():
                                plugin_commands_dir = install_path / "commands"
                                if plugin_commands_dir.exists():
                                    for cmd_file in plugin_commands_dir.glob("*.md"):
                                        command = self._parse_command_file(
                                            cmd_file, ConfigSource.PLUGIN
                                        )
                                        if command:
                                            commands.append(command)
            except (OSError, json.JSONDecodeError):
                # Failed to read plugin commands; skip
                pass

        return commands

    def _parse_command_file(self, cmd_file: Path, source: ConfigSource) -> Optional[Command]:
        """Parse a command file."""
        try:
            with open(cmd_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract description from first line or header
            description = ""
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    description = line[:200]  # First 200 chars
                    break
                elif line.startswith('# '):
                    description = line.lstrip('# ').strip()
                    break

            return Command(
                name=cmd_file.stem,
                content=content,
                source=source,
                path=cmd_file,
                description=description
            )
        except OSError:
            return None

    def _parse_plugins(self, user_claude_dir: Path) -> List[Plugin]:
        """Parse installed plugins."""
        plugins = []

        installed_plugins_file = user_claude_dir / "plugins" / "installed_plugins.json"
        if not installed_plugins_file.exists():
            return plugins

        try:
            with open(installed_plugins_file, 'r', encoding='utf-8') as f:
                plugins_data = json.load(f)

            enabled_plugins = {}
            if self.config_parser:
                enabled_plugins = self.config_parser.get_enabled_plugins()

            for plugin_name, versions in plugins_data.get('plugins', {}).items():
                if versions:
                    latest_version = versions[0]
                    install_path = Path(latest_version.get('installPath', ''))

                    # Read plugin metadata
                    plugin_json = install_path / ".claude-plugin" / "plugin.json"
                    description = ""
                    author = None
                    license_type = None
                    keywords = []

                    if plugin_json.exists():
                        try:
                            with open(plugin_json, 'r', encoding='utf-8') as pf:
                                metadata = json.load(pf)
                                description = metadata.get('description', '')
                                author = metadata.get('author', {}).get('name') if isinstance(
                                    metadata.get('author'), dict
                                ) else metadata.get('author')
                                license_type = metadata.get('license')
                                keywords = metadata.get('keywords', [])
                        except (OSError, json.JSONDecodeError):
                            # Failed to read plugin metadata; use defaults
                            pass

                    # Count features
                    features = {}
                    if (install_path / "skills").exists():
                        features['skills'] = len(list((install_path / "skills").iterdir()))
                    if (install_path / "commands").exists():
                        features['commands'] = len(list((install_path / "commands").glob("*.md")))
                    if (install_path / "agents").exists():
                        features['agents'] = len(list((install_path / "agents").glob("*.md")))
                    if (install_path / ".mcp.json").exists():
                        features['mcps'] = 1

                    plugins.append(Plugin(
                        name=plugin_name,
                        version=latest_version.get('version', 'unknown'),
                        description=description,
                        install_path=install_path,
                        author=author,
                        license=license_type,
                        keywords=keywords,
                        enabled=enabled_plugins.get(plugin_name, False),
                        features=features
                    ))
        except (OSError, json.JSONDecodeError):
            # Failed to read plugins list; return empty list
            pass

        return plugins

    def _parse_hooks(self, user_claude_dir: Path) -> List[Hook]:
        """Parse hooks from installed plugins."""
        hooks = []

        installed_plugins_file = user_claude_dir / "plugins" / "installed_plugins.json"
        if not installed_plugins_file.exists():
            return hooks

        try:
            with open(installed_plugins_file, 'r', encoding='utf-8') as f:
                plugins_data = json.load(f)

            for plugin_name, versions in plugins_data.get('plugins', {}).items():
                if versions:
                    latest_version = versions[0]
                    install_path = Path(latest_version.get('installPath', ''))
                    hooks_config_file = install_path / "hooks" / "hooks.json"

                    if hooks_config_file.exists():
                        try:
                            with open(hooks_config_file, 'r', encoding='utf-8') as hf:
                                hooks_config = json.load(hf)

                                for hook_type, hook_configs in hooks_config.get('hooks', {}).items():
                                    for hook_config in hook_configs:
                                        for hook_def in hook_config.get('hooks', []):
                                            handler_path = None
                                            if 'command' in hook_def:
                                                command_str = hook_def['command']
                                                # Try to extract path from command
                                                if '${CLAUDE_PLUGIN_ROOT}' in command_str:
                                                    rel_path = command_str.split('${CLAUDE_PLUGIN_ROOT}/')[-1]
                                                    handler_path = install_path / rel_path.split()[0]

                                            hooks.append(Hook(
                                                name=f"{plugin_name}_{hook_type}",
                                                hook_type=hook_type,
                                                triggers=[hook_type],
                                                matchers=hook_config.get('matcher'),
                                                handler_type=hook_def.get('type', 'command'),
                                                handler_path=handler_path,
                                                source=ConfigSource.PLUGIN,
                                                plugin_name=plugin_name
                                            ))
                        except (OSError, json.JSONDecodeError):
                            # Failed to read hooks config; skip this plugin
                            pass
        except (OSError, json.JSONDecodeError):
            # Failed to read installed plugins; return empty list
            pass

        return hooks

    def _parse_agents(self, project_claude_dir: Path, user_claude_dir: Path) -> List[Agent]:
        """Parse agents from user directory, project directory, and plugins."""
        agents = []

        # Parse user-level agents from ~/.claude/agents/
        user_agents_dir = user_claude_dir / "agents"
        if user_agents_dir.exists():
            for agent_file in user_agents_dir.glob("*.md"):
                agent = self._parse_agent_file(agent_file, ConfigSource.USER)
                if agent:
                    agents.append(agent)

        # Parse project-level agents from .claude/agents/
        project_agents_dir = project_claude_dir / "agents"
        if project_agents_dir.exists():
            for agent_file in project_agents_dir.glob("*.md"):
                agent = self._parse_agent_file(agent_file, ConfigSource.PROJECT)
                if agent:
                    agents.append(agent)

        # Parse plugin agents
        installed_plugins_file = user_claude_dir / "plugins" / "installed_plugins.json"
        if installed_plugins_file.exists():
            try:
                with open(installed_plugins_file, 'r', encoding='utf-8') as f:
                    plugins_data = json.load(f)

                for plugin_name, versions in plugins_data.get('plugins', {}).items():
                    if versions:
                        latest_version = versions[0]
                        install_path = Path(latest_version.get('installPath', ''))
                        agents_dir = install_path / "agents"

                        if agents_dir.exists():
                            for agent_file in agents_dir.glob("*.md"):
                                agent = self._parse_agent_file(agent_file, ConfigSource.PLUGIN)
                                if agent:
                                    agents.append(agent)
            except (OSError, json.JSONDecodeError):
                # Failed to read plugin agents; skip
                pass

        return agents

    def _parse_agent_file(self, agent_file: Path, source: ConfigSource) -> Optional[Agent]:
        """Parse an agent file."""
        try:
            with open(agent_file, 'r', encoding='utf-8') as f:
                content = f.read()

            description = ""
            model = "inherit"
            color = None
            tools = []

            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    frontmatter = parts[1]
                    for line in frontmatter.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip()
                            value = value.strip()
                            if key == 'description':
                                description = value
                            elif key == 'model':
                                model = value
                            elif key == 'color':
                                color = value
                            elif key == 'tools':
                                # Parse list
                                tools = [t.strip().strip('[]"') for t in value.split(',')]

            if not description:
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('---') and not line.startswith('#'):
                        description = line[:200]
                        break

            return Agent(
                name=agent_file.stem,
                description=description,
                source=source,
                path=agent_file,
                model=model,
                color=color,
                tools=tools,
                instructions=content
            )
        except OSError:
            return None

    def resolve_feature_inheritance(
        self, feature_type: str, feature_name: str, repo_path: Path
    ) -> InheritanceChain:
        """
        Resolve the inheritance chain for a feature.

        Args:
            feature_type: Type of feature (skill, command, etc.)
            feature_name: Name of the feature
            repo_path: Repository path

        Returns:
            InheritanceChain showing all levels
        """
        chain = InheritanceChain(feature_type=feature_type, feature_name=feature_name)

        # This is a simplified implementation
        # In a full implementation, you would check user, plugin, and project levels
        # For now, we'll just mark the first found as active

        features = self.get_all_features(repo_path)

        # Find the feature in the appropriate list
        feature_list = {
            'skill': features.skills,
            'command': features.commands,
            'mcp': features.mcps,
            'plugin': features.plugins,
            'hook': features.hooks,
            'agent': features.agents,
        }.get(feature_type, [])

        for feature in feature_list:
            if feature.name == feature_name:
                chain.levels.append(InheritanceLevel(
                    source=feature.source,
                    path=feature.path if hasattr(feature, 'path') else feature.install_path,
                    feature=feature
                ))
                break

        return chain
