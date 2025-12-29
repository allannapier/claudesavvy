"""Utilities for locating Claude Code data directories and files."""

from pathlib import Path
from typing import Optional


class ClaudeDataPaths:
    """Manages paths to Claude Code local data storage."""

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize Claude data paths.

        Args:
            base_dir: Optional custom base directory. Defaults to ~/.claude/
        """
        self.base_dir = base_dir or Path.home() / ".claude"

    def validate(self) -> bool:
        """Check if Claude data directory exists."""
        return self.base_dir.exists() and self.base_dir.is_dir()

    @property
    def history_file(self) -> Path:
        """Path to command history file (history.jsonl)."""
        return self.base_dir / "history.jsonl"

    @property
    def projects_dir(self) -> Path:
        """Path to projects directory containing session data."""
        return self.base_dir / "projects"

    @property
    def debug_dir(self) -> Path:
        """Path to debug logs directory."""
        return self.base_dir / "debug"

    @property
    def file_history_dir(self) -> Path:
        """Path to file history/backups directory."""
        return self.base_dir / "file-history"

    @property
    def todos_dir(self) -> Path:
        """Path to todos directory."""
        return self.base_dir / "todos"

    @property
    def settings_file(self) -> Path:
        """Path to settings.json file."""
        return self.base_dir / "settings.json"

    @property
    def shell_snapshots_dir(self) -> Path:
        """Path to shell snapshots directory."""
        return self.base_dir / "shell-snapshots"

    def get_project_session_files(self) -> list[Path]:
        """
        Get all session JSONL files from projects directory.

        Returns:
            List of paths to .jsonl session files
        """
        if not self.projects_dir.exists():
            return []

        session_files = []
        for project_dir in self.projects_dir.iterdir():
            if project_dir.is_dir():
                session_files.extend(project_dir.glob("*.jsonl"))

        return sorted(session_files, key=lambda p: p.stat().st_mtime, reverse=True)

    def get_debug_log_files(self) -> list[Path]:
        """
        Get all debug log files.

        Returns:
            List of paths to debug log files
        """
        if not self.debug_dir.exists():
            return []

        return sorted(
            self.debug_dir.glob("*.txt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )


def get_claude_paths() -> ClaudeDataPaths:
    """
    Get default Claude data paths instance.

    Returns:
        ClaudeDataPaths instance

    Raises:
        FileNotFoundError: If Claude data directory doesn't exist
    """
    paths = ClaudeDataPaths()
    if not paths.validate():
        raise FileNotFoundError(
            f"Claude data directory not found at {paths.base_dir}. "
            "Make sure Claude Code has been used at least once."
        )
    return paths
