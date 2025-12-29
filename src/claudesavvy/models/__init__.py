"""Models package for Claude Monitor."""

from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any


class ConfigSource(Enum):
    """Source of a configuration feature."""
    USER = "user"
    PLUGIN = "plugin"
    PROJECT = "project"


@dataclass
class RepositoryConfig:
    """Repository configuration information."""
    path: Path
    name: str
    claude_dir: Path

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'path': str(self.path),
            'name': self.name,
            'claude_dir': str(self.claude_dir),
        }


@dataclass
class Skill:
    """Claude Code skill configuration."""
    name: str
    description: str
    source: ConfigSource
    path: Path
    content: str = ""
    version: Optional[str] = None
    agents: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'source': self.source.value,
            'path': str(self.path),
            'content': self.content,
            'version': self.version,
            'agents': self.agents,
        }


@dataclass
class MCP:
    """Model Context Protocol server configuration."""
    name: str
    command: str
    args: List[str]
    source: ConfigSource
    plugin_path: Optional[Path] = None
    env: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'command': self.command,
            'args': self.args,
            'source': self.source.value,
            'plugin_path': str(self.plugin_path) if self.plugin_path else None,
            'env': self.env,
        }


@dataclass
class Command:
    """Slash command configuration."""
    name: str
    content: str
    source: ConfigSource
    path: Path
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'content': self.content,
            'source': self.source.value,
            'path': str(self.path),
            'description': self.description,
        }


@dataclass
class Plugin:
    """Plugin configuration."""
    name: str
    version: str
    description: str
    install_path: Path
    source: ConfigSource = ConfigSource.PLUGIN
    author: Optional[str] = None
    license: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    enabled: bool = True
    features: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'install_path': str(self.install_path),
            'source': self.source.value,
            'author': self.author,
            'license': self.license,
            'keywords': self.keywords,
            'enabled': self.enabled,
            'features': self.features,
        }


@dataclass
class Hook:
    """Hook configuration."""
    name: str
    hook_type: str
    triggers: List[str]
    matchers: Optional[List[str]]
    handler_type: str
    handler_path: Optional[Path]
    source: ConfigSource
    plugin_name: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.name,  # Use name as description for compatibility
            'hook_type': self.hook_type,
            'triggers': self.triggers,
            'matchers': self.matchers or [],
            'handler': str(self.handler_path) if self.handler_path else self.handler_type,
            'handler_type': self.handler_type,
            'handler_path': str(self.handler_path) if self.handler_path else None,
            'source': self.source.value,
            'plugin_name': self.plugin_name,
            'config': self.config,
        }


@dataclass
class Agent:
    """Agent configuration."""
    name: str
    description: str
    source: ConfigSource
    path: Path
    model: Optional[str] = None
    color: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    instructions: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'source': self.source.value,
            'path': str(self.path),
            'model': self.model,
            'color': self.color,
            'tools': self.tools,
            'instructions': self.instructions,
        }


@dataclass
class InheritanceLevel:
    """Single level in an inheritance chain."""
    source: ConfigSource
    path: Path
    feature: Any  # The actual feature object

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'source': self.source.value,
            'path': str(self.path),
            'feature': self.feature.to_dict() if hasattr(self.feature, 'to_dict') else str(self.feature),
        }


@dataclass
class InheritanceChain:
    """Full inheritance chain for a feature."""
    feature_type: str
    feature_name: str
    levels: List[InheritanceLevel] = field(default_factory=list)
    winner: Optional[ConfigSource] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'feature_type': self.feature_type,
            'feature_name': self.feature_name,
            'levels': [level.to_dict() for level in self.levels],
            'winner': self.winner.value if self.winner else None,
        }


@dataclass
class ConfigurationFeatures:
    """All configuration features for a repository."""
    skills: List[Skill] = field(default_factory=list)
    mcps: List[MCP] = field(default_factory=list)
    commands: List[Command] = field(default_factory=list)
    plugins: List[Plugin] = field(default_factory=list)
    hooks: List[Hook] = field(default_factory=list)
    agents: List[Agent] = field(default_factory=list)

    def get_feature_count(self) -> Dict[str, int]:
        """Get count of each feature type."""
        return {
            'skills': len(self.skills),
            'mcps': len(self.mcps),
            'commands': len(self.commands),
            'plugins': len(self.plugins),
            'hooks': len(self.hooks),
            'agents': len(self.agents),
        }


@dataclass
class Recommendation:
    """A single optimization recommendation for a project."""
    category: str  # 'mcp', 'model', 'config', 'cache', 'agent'
    severity: str  # 'high', 'medium', 'low', 'info'
    title: str
    description: str
    impact: str  # "Estimated savings: ~$X/month" or similar
    action_items: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'category': self.category,
            'severity': self.severity,
            'title': self.title,
            'description': self.description,
            'impact': self.impact,
            'action_items': self.action_items,
            'details': self.details,
        }


@dataclass
class ProjectAnalysis:
    """Complete optimization analysis for a project."""
    project_path: str
    project_name: str
    recommendations: List[Recommendation]
    metrics: Dict[str, Any]
    analyzed_at: datetime

    @property
    def high_severity_count(self) -> int:
        """Count of high severity recommendations."""
        return sum(1 for r in self.recommendations if r.severity == 'high')

    @property
    def medium_severity_count(self) -> int:
        """Count of medium severity recommendations."""
        return sum(1 for r in self.recommendations if r.severity == 'medium')

    @property
    def low_severity_count(self) -> int:
        """Count of low severity recommendations."""
        return sum(1 for r in self.recommendations if r.severity == 'low')

    @property
    def total_recommendations(self) -> int:
        """Total count of recommendations."""
        return len(self.recommendations)

    def recommendations_by_category(self, category: str) -> List[Recommendation]:
        """Get recommendations filtered by category."""
        return [r for r in self.recommendations if r.category == category]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'project_path': self.project_path,
            'project_name': self.project_name,
            'recommendations': [r.to_dict() for r in self.recommendations],
            'metrics': self.metrics,
            'analyzed_at': self.analyzed_at.isoformat() if self.analyzed_at else None,
            'summary': {
                'high_severity': self.high_severity_count,
                'medium_severity': self.medium_severity_count,
                'low_severity': self.low_severity_count,
                'total': self.total_recommendations,
            }
        }


__all__ = [
    'ConfigSource',
    'RepositoryConfig',
    'Skill',
    'MCP',
    'Command',
    'Plugin',
    'Hook',
    'Agent',
    'InheritanceLevel',
    'InheritanceChain',
    'ConfigurationFeatures',
    'Recommendation',
    'ProjectAnalysis',
]
