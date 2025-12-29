"""Parser for Claude Code command history (history.jsonl)."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from ..utils.time_filter import TimeFilter


@dataclass
class HistoryEntry:
    """Represents a single command from history."""

    display: str
    timestamp: int  # milliseconds since epoch
    project: Optional[str]
    pasted_contents: dict

    @property
    def datetime(self) -> datetime:
        """Get datetime from timestamp."""
        return datetime.fromtimestamp(self.timestamp / 1000)

    @classmethod
    def from_dict(cls, data: dict) -> "HistoryEntry":
        """Create HistoryEntry from parsed JSON dict."""
        return cls(
            display=data.get("display", ""),
            timestamp=data.get("timestamp", 0),
            project=data.get("project"),
            pasted_contents=data.get("pastedContents", {})
        )


class HistoryParser:
    """Parser for Claude Code command history."""

    def __init__(self, history_file: Path):
        """
        Initialize history parser.

        Args:
            history_file: Path to history.jsonl file
        """
        self.history_file = history_file

    def parse(
        self,
        time_filter: Optional[TimeFilter] = None,
        project_filter: Optional[str] = None
    ) -> Iterator[HistoryEntry]:
        """
        Parse history file and yield entries.

        Args:
            time_filter: Optional time filter
            project_filter: Optional project path to filter by

        Yields:
            HistoryEntry instances
        """
        if not self.history_file.exists():
            return

        with open(self.history_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    entry = HistoryEntry.from_dict(data)

                    # Apply filters
                    if time_filter and not time_filter.matches_timestamp_ms(entry.timestamp):
                        continue

                    if project_filter and entry.project != project_filter:
                        continue

                    yield entry

                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue

    def get_all_entries(
        self,
        time_filter: Optional[TimeFilter] = None,
        project_filter: Optional[str] = None
    ) -> list[HistoryEntry]:
        """
        Get all history entries as a list.

        Args:
            time_filter: Optional time filter
            project_filter: Optional project path to filter by

        Returns:
            List of HistoryEntry instances
        """
        return list(self.parse(time_filter=time_filter, project_filter=project_filter))

    def get_project_counts(
        self,
        time_filter: Optional[TimeFilter] = None
    ) -> dict[str, int]:
        """
        Get command counts per project.

        Args:
            time_filter: Optional time filter

        Returns:
            Dict mapping project paths to command counts
        """
        counts: dict[str, int] = {}

        for entry in self.parse(time_filter=time_filter):
            if entry.project:
                counts[entry.project] = counts.get(entry.project, 0) + 1

        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def get_command_count(
        self,
        time_filter: Optional[TimeFilter] = None,
        project_filter: Optional[str] = None
    ) -> int:
        """
        Get total command count.

        Args:
            time_filter: Optional time filter
            project_filter: Optional project path to filter by

        Returns:
            Total number of commands
        """
        return sum(1 for _ in self.parse(time_filter=time_filter, project_filter=project_filter))

    def get_date_range(self) -> tuple[Optional[datetime], Optional[datetime]]:
        """
        Get the date range of all history entries.

        Returns:
            Tuple of (earliest_date, latest_date)
        """
        entries = self.get_all_entries()
        if not entries:
            return None, None

        timestamps = [entry.datetime for entry in entries]
        return min(timestamps), max(timestamps)
