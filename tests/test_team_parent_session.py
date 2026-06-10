"""Tests for parent_session_id derivation in SubAgentParser and get_team_summary exposure."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from claudesavvy.parsers.sessions import SubAgentExchange, SubAgentParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_teammate_jsonl(path: Path, session_id: str, slug: str = "brave-tiger") -> None:
    """Write a minimal valid teammate agent JSONL file."""
    lines = [
        {
            "timestamp": "2024-06-01T10:00:00Z",
            "sessionId": session_id,
            "slug": slug,
            "cwd": "/tmp/project",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": '<teammate-message summary="Do some work" teammate_id="researcher">task</teammate-message>',
                    }
                ],
            },
        },
        {
            "timestamp": "2024-06-01T10:00:05Z",
            "sessionId": session_id,
            "slug": slug,
            "cwd": "/tmp/project",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
                "content": [],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Test: parent_session_id derivation from file path
# ---------------------------------------------------------------------------

def test_parent_session_id_derived_from_path(tmp_path):
    """parent_session_id should equal the grandparent dir name when parent dir is 'subagents'."""
    parent_uuid = "aaaabbbb-1111-2222-3333-ccccddddeeee"
    agent_id = "agent-xyz"
    teammate_session_id = "tttt-0000-1111-2222-333344445555"

    subagents_dir = tmp_path / "my-project" / parent_uuid / "subagents"
    subagents_dir.mkdir(parents=True)
    agent_file = subagents_dir / f"{agent_id}.jsonl"
    _write_teammate_jsonl(agent_file, session_id=teammate_session_id)

    parser = SubAgentParser(
        session_files=[],
        subagent_file_map={agent_id: agent_file},
    )
    exchanges = parser.parse_exchanges()

    assert len(exchanges) == 1
    ex = exchanges[0]
    assert ex.is_teammate is True
    assert ex.parent_session_id == parent_uuid


def test_parent_session_id_empty_when_not_in_subagents_dir(tmp_path):
    """parent_session_id should be empty when the file is not inside a 'subagents' directory."""
    agent_id = "agent-abc"
    teammate_session_id = "tttt-0000-aaaa-bbbb-ccccddddeeee"

    # Flat structure: no 'subagents' parent dir
    flat_dir = tmp_path / "some-parent"
    flat_dir.mkdir(parents=True)
    agent_file = flat_dir / f"{agent_id}.jsonl"
    _write_teammate_jsonl(agent_file, session_id=teammate_session_id)

    parser = SubAgentParser(
        session_files=[],
        subagent_file_map={agent_id: agent_file},
    )
    exchanges = parser.parse_exchanges()

    assert len(exchanges) == 1
    assert exchanges[0].parent_session_id == ""


# ---------------------------------------------------------------------------
# Test: get_team_summary exposes parent_session_id
# ---------------------------------------------------------------------------

def test_get_team_summary_includes_parent_session_id():
    """get_team_summary should surface parent_session_id from the first exchange that has one."""
    from claudesavvy.web.services.dashboard_service import DashboardService

    parent_uuid = "ffff1111-aaaa-bbbb-cccc-ddddeeee0000"
    teammate_session_id = "sess-0000-1111-2222-333344445555"

    exchanges = [
        SubAgentExchange(
            agent_id="agent-1",
            session_id=teammate_session_id,
            project="/tmp/project",
            timestamp="2024-06-01T10:00:00Z",
            subagent_type="researcher",
            slug="brave-tiger",
            is_teammate=True,
            parent_session_id=parent_uuid,
            total_tokens=100,
        ),
        SubAgentExchange(
            agent_id="agent-2",
            session_id=teammate_session_id,
            project="/tmp/project",
            timestamp="2024-06-01T10:00:01Z",
            subagent_type="writer",
            slug="brave-tiger",
            is_teammate=True,
            parent_session_id=parent_uuid,
            total_tokens=80,
        ),
    ]

    service = DashboardService.__new__(DashboardService)
    mock_parser = MagicMock()
    mock_parser.parse_exchanges.return_value = exchanges
    service._subagent_parser = mock_parser

    result = service.get_team_summary()

    assert result["total_teams"] == 1
    team = result["teams"][0]
    assert team["parent_session_id"] == parent_uuid


def test_get_team_summary_parent_session_id_empty_when_absent():
    """parent_session_id should be empty string when no exchange has one."""
    from claudesavvy.web.services.dashboard_service import DashboardService

    teammate_session_id = "sess-zzzz-1111-2222-333344445555"

    exchanges = [
        SubAgentExchange(
            agent_id="agent-3",
            session_id=teammate_session_id,
            project="/tmp/project",
            timestamp="2024-06-01T11:00:00Z",
            subagent_type="researcher",
            slug="calm-whale",
            is_teammate=True,
            parent_session_id="",
            total_tokens=50,
        ),
    ]

    service = DashboardService.__new__(DashboardService)
    mock_parser = MagicMock()
    mock_parser.parse_exchanges.return_value = exchanges
    service._subagent_parser = mock_parser

    result = service.get_team_summary()

    assert result["teams"][0]["parent_session_id"] == ""


# ---------------------------------------------------------------------------
# Test: harness_grade / harness_score included in get_team_summary
# ---------------------------------------------------------------------------

def test_get_team_summary_includes_harness_grade_when_scored():
    """harness_grade and harness_score should be populated when _get_session_grade returns a value."""
    from claudesavvy.web.services.dashboard_service import DashboardService

    parent_uuid = "aaaa1111-bbbb-cccc-dddd-eeee00001111"
    teammate_session_id = "sess-1111-2222-3333-444455556666"

    exchanges = [
        SubAgentExchange(
            agent_id="agent-10",
            session_id=teammate_session_id,
            project="/tmp/project",
            timestamp="2024-06-01T12:00:00Z",
            subagent_type="researcher",
            slug="swift-fox",
            is_teammate=True,
            parent_session_id=parent_uuid,
            total_tokens=200,
        ),
    ]

    service = DashboardService.__new__(DashboardService)
    service._session_grade_cache = {}
    mock_parser = MagicMock()
    mock_parser.parse_exchanges.return_value = exchanges
    service._subagent_parser = mock_parser
    service._get_session_grade = lambda session_id: {"grade": "B", "score": 81}

    result = service.get_team_summary()

    team = result["teams"][0]
    assert team["harness_grade"] == "B"
    assert team["harness_score"] == 81


def test_get_team_summary_harness_none_when_no_parent_session_id():
    """harness_grade/score should be None and _get_session_grade should not be called when parent_session_id is empty."""
    from claudesavvy.web.services.dashboard_service import DashboardService

    teammate_session_id = "sess-9999-aaaa-bbbb-ccccddddeeee"

    exchanges = [
        SubAgentExchange(
            agent_id="agent-11",
            session_id=teammate_session_id,
            project="/tmp/project",
            timestamp="2024-06-01T13:00:00Z",
            subagent_type="writer",
            slug="quiet-owl",
            is_teammate=True,
            parent_session_id="",
            total_tokens=60,
        ),
    ]

    service = DashboardService.__new__(DashboardService)
    service._session_grade_cache = {}
    mock_parser = MagicMock()
    mock_parser.parse_exchanges.return_value = exchanges
    service._subagent_parser = mock_parser

    called = []
    service._get_session_grade = lambda session_id: called.append(session_id) or {}

    result = service.get_team_summary()

    assert called == [], "_get_session_grade should not be called when parent_session_id is empty"
    team = result["teams"][0]
    assert team["harness_grade"] is None
    assert team["harness_score"] is None


# ---------------------------------------------------------------------------
# Test: _get_session_grade caching
# ---------------------------------------------------------------------------

def test_get_session_grade_caches_result(tmp_path):
    """evaluate_file should only be called once for repeated _get_session_grade calls."""
    import claudesavvy.web.services.dashboard_service as ds_module
    from claudesavvy.web.services.dashboard_service import DashboardService

    session_id = "abcd1234-ef56-7890-abcd-ef1234567890"
    session_file = tmp_path / f"{session_id}.jsonl"
    session_file.write_text("{}\n", encoding="utf-8")

    service = DashboardService.__new__(DashboardService)
    service._session_grade_cache = {}

    mock_paths = MagicMock()
    mock_paths.get_project_session_files.return_value = [session_file]
    service.paths = mock_paths

    call_count = []

    def fake_evaluate_file(_path):
        call_count.append(1)
        return {"grade": "A", "score": 95}

    with patch.object(ds_module.harness_quality, "evaluate_file", side_effect=fake_evaluate_file):
        result1 = service._get_session_grade(session_id)
        result2 = service._get_session_grade(session_id)

    assert result1 == {"grade": "A", "score": 95}
    assert result2 == {"grade": "A", "score": 95}
    assert len(call_count) == 1, f"evaluate_file called {len(call_count)} times, expected 1"
