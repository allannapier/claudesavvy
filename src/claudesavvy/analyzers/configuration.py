"""Analyzer for Claude Code configuration data."""

from pathlib import Path
from typing import Dict, Any, List
from ..parsers.configuration_scanner import ConfigurationScanner


class ConfigurationAnalyzer:
    """Analyzes Claude Code configuration data."""

    def __init__(self, scanner: ConfigurationScanner):
        """
        Initialize configuration analyzer.

        Args:
            scanner: ConfigurationScanner instance
        """
        self.scanner = scanner

    def get_repository_summary(self, repositories: List) -> Dict[str, Any]:
        """
        Get summary statistics for all repositories.

        Args:
            repositories: List of RepositoryConfig objects

        Returns:
            Dict with summary statistics
        """
        total_repos = len(repositories)
        total_features = {
            'skills': 0,
            'mcps': 0,
            'commands': 0,
            'plugins': 0,
            'hooks': 0,
            'agents': 0
        }

        for repo in repositories:
            features = self.scanner.get_all_features(repo.path)
            counts = features.get_feature_count()
            for key in total_features:
                total_features[key] += counts.get(key, 0)

        return {
            'total_repositories': total_repos,
            'total_features': total_features,
            'repositories': [repo.to_dict() for repo in repositories]
        }

    def get_feature_breakdown(self, repo_path: Path) -> Dict[str, List[Dict]]:
        """
        Get all features for a repository, broken down by category.

        Args:
            repo_path: Path to repository

        Returns:
            Dict mapping feature type to list of features
        """
        features = self.scanner.get_all_features(repo_path)

        return {
            'skills': [skill.to_dict() for skill in features.skills],
            'mcps': [mcp.to_dict() for mcp in features.mcps],
            'commands': [cmd.to_dict() for cmd in features.commands],
            'plugins': [plugin.to_dict() for plugin in features.plugins],
            'hooks': [hook.to_dict() for hook in features.hooks],
            'agents': [agent.to_dict() for agent in features.agents],
            'counts': features.get_feature_count()
        }

    def get_inheritance_tree(
        self, feature_type: str, feature_name: str, repo_path: Path
    ) -> Dict[str, Any]:
        """
        Get inheritance visualization data for a feature.

        Args:
            feature_type: Type of feature
            feature_name: Name of feature
            repo_path: Repository path

        Returns:
            Dict with inheritance tree data
        """
        chain = self.scanner.resolve_feature_inheritance(
            feature_type, feature_name, repo_path
        )

        return chain.to_dict()

    def get_feature_detail(
        self, feature_type: str, feature_name: str, repo_path: Path
    ) -> Dict[str, Any]:
        """
        Get full details for a specific feature.

        Args:
            feature_type: Type of feature
            feature_name: Name of feature
            repo_path: Repository path

        Returns:
            Dict with feature details including content
        """
        features = self.scanner.get_all_features(repo_path)

        # Find the feature
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
                detail = feature.to_dict()

                # Add full content for features that have it
                if hasattr(feature, 'content') and feature.content:
                    detail['full_content'] = feature.content

                # Add inheritance chain
                detail['inheritance'] = self.get_inheritance_tree(
                    feature_type, feature_name, repo_path
                )

                return detail

        return {}

    def get_feature_conflicts(self, repo_path: Path) -> List[Dict[str, Any]]:
        """
        Identify features that have conflicts or overrides.

        Args:
            repo_path: Repository path

        Returns:
            List of conflicts/overrides
        """
        conflicts = []
        features = self.scanner.get_all_features(repo_path)

        # Check for duplicate feature names across sources
        # This is a simplified implementation
        skill_names = {}
        for skill in features.skills:
            if skill.name in skill_names:
                conflicts.append({
                    'type': 'skill',
                    'name': skill.name,
                    'sources': [skill_names[skill.name].value, skill.source.value],
                    'message': f'Skill "{skill.name}" defined in multiple locations'
                })
            else:
                skill_names[skill.name] = skill.source

        command_names = {}
        for command in features.commands:
            if command.name in command_names:
                conflicts.append({
                    'type': 'command',
                    'name': command.name,
                    'sources': [command_names[command.name].value, command.source.value],
                    'message': f'Command "{command.name}" defined in multiple locations'
                })
            else:
                command_names[command.name] = command.source

        return conflicts
