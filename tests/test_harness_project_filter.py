"""Tests for harness project filtering in DashboardService."""

from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_record(project: str) -> dict:
    """Minimal evaluate_file record covering all keys accessed by get_harness_evaluation."""
    return {
        "session_id": f"sess-{project}",
        "project": project,
        "project_label": project,
        "score": 80,
        "grade": "B",
        "low_activity": False,
        "subscores": {
            "duplicate_commands": 100,
            "repeated_reads": 100,
            "scriptability": 100,
        },
        "metrics": {
            "tool_errors": 0,
            "user_rejections": 0,
        },
    }


def _make_service():
    """Create a DashboardService instance without running __init__."""
    from claudesavvy.web.services.dashboard_service import DashboardService

    service = DashboardService.__new__(DashboardService)
    service.paths = MagicMock()
    return service


def _mock_session_files(tmp_path: Path):
    """Create two session files in proj-a and proj-b dirs."""
    proj_a = tmp_path / "proj-a"
    proj_b = tmp_path / "proj-b"
    proj_a.mkdir()
    proj_b.mkdir()
    file_a = proj_a / "session-a.jsonl"
    file_b = proj_b / "session-b.jsonl"
    file_a.write_text("")
    file_b.write_text("")
    return file_a, file_b


class TestGetHarnessEvaluationProjectFilter:
    def test_filters_to_project(self, tmp_path):
        file_a, file_b = _mock_session_files(tmp_path)
        service = _make_service()
        service.paths.get_project_session_files.return_value = [file_a, file_b]

        with patch(
            "claudesavvy.web.services.dashboard_service.harness_quality.evaluate_file",
            side_effect=lambda p: _make_record(p.parent.name),
        ):
            result = service.get_harness_evaluation(project="proj-a")

        assert result["session_count"] == 1
        assert result["sessions"][0]["project"] == "proj-a"

    def test_no_project_includes_all(self, tmp_path):
        file_a, file_b = _mock_session_files(tmp_path)
        service = _make_service()
        service.paths.get_project_session_files.return_value = [file_a, file_b]

        with patch(
            "claudesavvy.web.services.dashboard_service.harness_quality.evaluate_file",
            side_effect=lambda p: _make_record(p.parent.name),
        ):
            result = service.get_harness_evaluation(project=None)

        assert result["session_count"] == 2

    def test_nonexistent_project_returns_empty(self, tmp_path):
        file_a, file_b = _mock_session_files(tmp_path)
        service = _make_service()
        service.paths.get_project_session_files.return_value = [file_a, file_b]

        with patch(
            "claudesavvy.web.services.dashboard_service.harness_quality.evaluate_file",
            side_effect=lambda p: _make_record(p.parent.name),
        ):
            result = service.get_harness_evaluation(project="proj-c")

        assert result["session_count"] == 0


class TestGetHarnessProjects:
    def test_returns_unique_sorted_entries(self, tmp_path):
        file_a, file_b = _mock_session_files(tmp_path)
        # Add a second file in proj-a to confirm deduplication
        extra = tmp_path / "proj-a" / "session-a2.jsonl"
        extra.write_text("")

        service = _make_service()
        service.paths.get_project_session_files.return_value = [file_a, extra, file_b]

        result = service.get_harness_projects()

        values = [e["value"] for e in result]
        assert values.count("proj-a") == 1
        assert "proj-b" in values
        # Sorted by label ascending
        labels = [e["label"] for e in result]
        assert labels == sorted(labels, key=str.lower)

    def test_has_value_and_label_keys(self, tmp_path):
        file_a, _ = _mock_session_files(tmp_path)
        service = _make_service()
        service.paths.get_project_session_files.return_value = [file_a]

        result = service.get_harness_projects()

        assert len(result) == 1
        assert "value" in result[0]
        assert "label" in result[0]
        assert result[0]["value"] == "proj-a"
