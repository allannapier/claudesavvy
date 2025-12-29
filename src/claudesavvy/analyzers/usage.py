"""Usage analyzer for Claude Code activity metrics."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ..parsers.history import HistoryParser
from ..parsers.sessions import SessionParser
from ..utils.time_filter import TimeFilter


@dataclass
class UsageSummary:
    """Summary of Claude Code usage."""

    total_commands: int
    total_sessions: int
    total_messages: int
    active_projects: int
    earliest_activity: Optional[datetime]
    latest_activity: Optional[datetime]
    time_range_description: str

    @property
    def date_range_days(self) -> Optional[int]:
        """Calculate the date range in days."""
        if self.earliest_activity and self.latest_activity:
            delta = self.latest_activity - self.earliest_activity
            return delta.days
        return None

    @property
    def avg_commands_per_day(self) -> Optional[float]:
        """Calculate average commands per day."""
        days = self.date_range_days
        if days and days > 0:
            return self.total_commands / days
        return None

    @property
    def avg_sessions_per_day(self) -> Optional[float]:
        """Calculate average sessions per day."""
        days = self.date_range_days
        if days and days > 0:
            return self.total_sessions / days
        return None


class UsageAnalyzer:
    """Analyzer for Claude Code usage patterns."""

    def __init__(
        self,
        history_parser: HistoryParser,
        session_parser: SessionParser,
        time_filter: Optional[TimeFilter] = None
    ):
        """
        Initialize usage analyzer.

        Args:
            history_parser: Parser for command history
            session_parser: Parser for session data
            time_filter: Optional time filter
        """
        self.history_parser = history_parser
        self.session_parser = session_parser
        self.time_filter = time_filter

    def get_summary(self) -> UsageSummary:
        """
        Get usage summary statistics.

        Returns:
            UsageSummary with aggregated metrics
        """
        # Get command count and date range from history
        total_commands = self.history_parser.get_command_count(time_filter=self.time_filter)
        earliest, latest = self.history_parser.get_date_range()

        # If time filter is applied, adjust date range
        if self.time_filter:
            if self.time_filter.start_time:
                earliest = max(earliest, self.time_filter.start_time) if earliest else self.time_filter.start_time
            if self.time_filter.end_time:
                latest = min(latest, self.time_filter.end_time) if latest else self.time_filter.end_time

        # Get session statistics
        session_stats = self.session_parser.get_stats(time_filter=self.time_filter)

        # Get time range description
        time_desc = self.time_filter.get_description() if self.time_filter else "All time"

        return UsageSummary(
            total_commands=total_commands,
            total_sessions=session_stats.session_count,
            total_messages=session_stats.message_count,
            active_projects=session_stats.project_count,
            earliest_activity=earliest,
            latest_activity=latest,
            time_range_description=time_desc
        )

    def get_project_breakdown(self) -> dict[str, dict]:
        """
        Get per-project breakdown of activity.

        Returns:
            Dict mapping project paths to activity metrics
        """
        # Get command counts per project
        command_counts = self.history_parser.get_project_counts(time_filter=self.time_filter)

        # Get session stats per project
        session_stats = self.session_parser.get_project_stats(time_filter=self.time_filter)

        # Combine into unified breakdown
        all_projects = set(command_counts.keys()) | set(session_stats.keys())

        breakdown = {}
        for project in all_projects:
            # Shorten project path for display
            short_name = project.split('/')[-1] if '/' in project else project

            breakdown[project] = {
                'name': short_name,
                'full_path': project,
                'commands': command_counts.get(project, 0),
                'sessions': session_stats[project].session_count if project in session_stats else 0,
                'messages': session_stats[project].message_count if project in session_stats else 0,
            }

        # Sort by command count (descending)
        return dict(
            sorted(breakdown.items(), key=lambda x: x[1]['commands'], reverse=True)
        )
