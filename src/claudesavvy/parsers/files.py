"""Parser for Claude Code file history/backups."""

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class FileStats:
    """Statistics for a single file."""

    file_path: str
    version_count: int
    total_size: int  # in bytes
    session_id: str

    @property
    def size_kb(self) -> float:
        """Size in kilobytes."""
        return self.total_size / 1024


class FileHistoryParser:
    """Parser for Claude Code file history."""

    # Pattern to extract version number from filename (e.g., "file@v1", "file@v2")
    VERSION_PATTERN = re.compile(r'@v(\d+)$')

    def __init__(self, file_history_dir: Path, session_files: Optional[list[Path]] = None):
        """
        Initialize file history parser.

        Args:
            file_history_dir: Path to file-history directory
            session_files: Optional list of session files to parse for file mappings
        """
        self.file_history_dir = file_history_dir
        self.session_files = session_files or []
        self._backup_to_path_map = self._build_backup_mapping()

    def _build_backup_mapping(self) -> dict[str, str]:
        """
        Build mapping from backup filenames to real file paths.

        Returns:
            Dict mapping backup filenames (without version) to real file paths
        """
        mapping = {}

        for session_file in self.session_files:
            if not session_file.exists():
                continue

            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if data.get('type') == 'file-history-snapshot':
                                snapshot = data.get('snapshot', {})
                                tracked_files = snapshot.get('trackedFileBackups', {})

                                for file_path, info in tracked_files.items():
                                    backup_filename = info.get('backupFileName', '')
                                    # Remove version suffix to get base backup name
                                    base_backup_name = self.VERSION_PATTERN.sub('', backup_filename)
                                    if base_backup_name:
                                        mapping[base_backup_name] = file_path
                        except (json.JSONDecodeError, KeyError):
                            continue
            except Exception:
                continue

        return mapping

    def get_session_stats(self, session_id: Optional[str] = None) -> dict[str, FileStats]:
        """
        Get file modification statistics, optionally filtered by session.

        Args:
            session_id: Optional session ID to filter by

        Returns:
            Dict mapping file paths to FileStats
        """
        stats: dict[str, FileStats] = {}

        if not self.file_history_dir.exists():
            return stats

        # Iterate through session directories
        for session_dir in self.file_history_dir.iterdir():
            if not session_dir.is_dir():
                continue

            # Filter by session if specified
            if session_id and session_dir.name != session_id:
                continue

            # Count versions for each file
            file_versions: dict[str, list[Path]] = defaultdict(list)

            for version_file in session_dir.iterdir():
                if version_file.is_file():
                    # Extract base filename (remove version suffix)
                    base_name = self.VERSION_PATTERN.sub('', version_file.name)
                    file_versions[base_name].append(version_file)

            # Create stats for each file
            for backup_name, versions in file_versions.items():
                # Look up real file path from backup name
                real_file_path = self._backup_to_path_map.get(backup_name)

                # Skip files we can't map to real paths (unknown hash IDs)
                if not real_file_path:
                    continue

                total_size = sum(v.stat().st_size for v in versions)

                stats[real_file_path] = FileStats(
                    file_path=real_file_path,
                    version_count=len(versions),
                    total_size=total_size,
                    session_id=session_dir.name
                )

        return stats

    def get_all_stats(self) -> dict[str, FileStats]:
        """
        Get aggregated file statistics across all sessions.

        Returns:
            Dict mapping file paths to aggregated FileStats
        """
        return self.get_session_stats()

    def get_most_edited_files(self, limit: int = 10) -> list[FileStats]:
        """
        Get the most frequently edited files.

        Args:
            limit: Maximum number of files to return

        Returns:
            List of FileStats sorted by version count (descending)
        """
        all_stats = self.get_all_stats()

        # Aggregate by file path across sessions
        aggregated: dict[str, FileStats] = {}

        for file_path, stats in all_stats.items():
            if file_path not in aggregated:
                aggregated[file_path] = FileStats(
                    file_path=file_path,
                    version_count=0,
                    total_size=0,
                    session_id="multiple"
                )

            aggregated[file_path].version_count += stats.version_count
            aggregated[file_path].total_size += stats.total_size

        # Sort by version count
        sorted_files = sorted(
            aggregated.values(),
            key=lambda x: x.version_count,
            reverse=True
        )

        return sorted_files[:limit]

    def get_total_file_count(self) -> int:
        """
        Get total number of unique files modified.

        Returns:
            Total file count
        """
        all_stats = self.get_all_stats()
        return len(set(s.file_path for s in all_stats.values()))

    def get_total_versions(self) -> int:
        """
        Get total number of file versions across all files.

        Returns:
            Total version count
        """
        all_stats = self.get_all_stats()
        return sum(s.version_count for s in all_stats.values())
