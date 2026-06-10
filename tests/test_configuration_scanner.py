"""Tests for ConfigurationScanner project-level parsing and harness totals fix.

Covers:
  (a) _parse_skills picks up project-level items with source "project"
  (b) _parse_commands picks up project-level items with source "project"
  (c) User-config-dir guard: no duplicates when project_claude_dir == user_claude_dir
  (d) harness _project_scoped helper filters correctly
  (e) harness totals don't multiply user items across repos
"""

from pathlib import Path

from claudesavvy.models import ConfigSource
from claudesavvy.parsers.configuration_scanner import ConfigurationScanner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skill_dir(parent: Path, name: str) -> Path:
    skill_dir = parent / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"# {name}\nA test skill.\n")
    return skill_dir


def _make_command_file(commands_dir: Path, name: str) -> Path:
    commands_dir.mkdir(parents=True, exist_ok=True)
    cmd_file = commands_dir / f"{name}.md"
    cmd_file.write_text(f"# {name}\nA test command.\n")
    return cmd_file


# ---------------------------------------------------------------------------
# (a) _parse_skills — project items get source PROJECT
# ---------------------------------------------------------------------------

class TestParseSkillsProject:
    def test_project_skill_has_project_source(self, tmp_path):
        user_claude = tmp_path / "user_claude"
        project_claude = tmp_path / "project_claude"

        # User skill
        _make_skill_dir(user_claude / "skills", "user-skill")
        # Project skill
        _make_skill_dir(project_claude / "skills", "project-skill")

        scanner = ConfigurationScanner()
        skills = scanner._parse_skills(project_claude, user_claude)

        sources = {s.name: s.source for s in skills}
        assert "user-skill" in sources
        assert sources["user-skill"] == ConfigSource.USER
        assert "project-skill" in sources
        assert sources["project-skill"] == ConfigSource.PROJECT

    def test_project_skills_dir_missing_is_noop(self, tmp_path):
        user_claude = tmp_path / "user_claude"
        project_claude = tmp_path / "project_claude"

        _make_skill_dir(user_claude / "skills", "user-skill")
        # project_claude has no skills/ directory

        scanner = ConfigurationScanner()
        skills = scanner._parse_skills(project_claude, user_claude)

        assert len(skills) == 1
        assert skills[0].name == "user-skill"

    def test_project_skill_not_parsed_when_no_skill_md(self, tmp_path):
        user_claude = tmp_path / "user_claude"
        project_claude = tmp_path / "project_claude"

        # skill dir with no SKILL.md / SKILLS.md
        bad_dir = project_claude / "skills" / "empty-skill"
        bad_dir.mkdir(parents=True)

        scanner = ConfigurationScanner()
        skills = scanner._parse_skills(project_claude, user_claude)
        # Should not include the empty directory
        assert all(s.source == ConfigSource.USER for s in skills)


# ---------------------------------------------------------------------------
# (b) _parse_commands — project items get source PROJECT
# ---------------------------------------------------------------------------

class TestParseCommandsProject:
    def test_project_command_has_project_source(self, tmp_path):
        user_claude = tmp_path / "user_claude"
        project_claude = tmp_path / "project_claude"

        _make_command_file(user_claude / "commands", "user-cmd")
        _make_command_file(project_claude / "commands", "project-cmd")

        scanner = ConfigurationScanner()
        commands = scanner._parse_commands(project_claude, user_claude)

        sources = {c.name: c.source for c in commands}
        assert "user-cmd" in sources
        assert sources["user-cmd"] == ConfigSource.USER
        assert "project-cmd" in sources
        assert sources["project-cmd"] == ConfigSource.PROJECT

    def test_project_commands_dir_missing_is_noop(self, tmp_path):
        user_claude = tmp_path / "user_claude"
        project_claude = tmp_path / "project_claude"

        _make_command_file(user_claude / "commands", "user-cmd")

        scanner = ConfigurationScanner()
        commands = scanner._parse_commands(project_claude, user_claude)

        assert len(commands) == 1
        assert commands[0].name == "user-cmd"


# ---------------------------------------------------------------------------
# (c) User-config-dir guard — no duplicates when project_claude_dir == user_claude_dir
# ---------------------------------------------------------------------------

class TestUserConfigDirGuard:
    def test_skills_not_doubled_when_project_equals_user(self, tmp_path):
        user_claude = tmp_path / "dot_claude"
        _make_skill_dir(user_claude / "skills", "my-skill")

        scanner = ConfigurationScanner()
        # Both arguments point to the same directory (as happens for the ~/.claude repo)
        skills = scanner._parse_skills(user_claude, user_claude)

        names = [s.name for s in skills]
        assert names.count("my-skill") == 1, "skill must not be parsed twice"
        assert skills[0].source == ConfigSource.USER

    def test_commands_not_doubled_when_project_equals_user(self, tmp_path):
        user_claude = tmp_path / "dot_claude"
        _make_command_file(user_claude / "commands", "my-cmd")

        scanner = ConfigurationScanner()
        commands = scanner._parse_commands(user_claude, user_claude)

        names = [c.name for c in commands]
        assert names.count("my-cmd") == 1, "command must not be parsed twice"
        assert commands[0].source == ConfigSource.USER

    def test_guard_uses_resolved_path(self, tmp_path):
        """Symlink to the same dir must still trigger the guard."""
        user_claude = tmp_path / "dot_claude"
        user_claude.mkdir()
        (user_claude / "skills").mkdir()
        _make_skill_dir(user_claude / "skills", "sym-skill")

        symlink_claude = tmp_path / "symlinked_claude"
        symlink_claude.symlink_to(user_claude)

        scanner = ConfigurationScanner()
        skills = scanner._parse_skills(symlink_claude, user_claude)

        names = [s.name for s in skills]
        assert names.count("sym-skill") == 1


# ---------------------------------------------------------------------------
# (d) _project_scoped helper (imported from dashboard module)
# ---------------------------------------------------------------------------

class TestProjectScopedHelper:
    def _get_helper(self):
        # Import the module-level helper defined in dashboard.py
        from claudesavvy.web.routes import dashboard as dash_mod
        return dash_mod._project_scoped

    def test_filters_to_project_source_only(self):
        _project_scoped = self._get_helper()
        features = {
            "skills": [
                {"name": "user-skill", "source": "user"},
                {"name": "plugin-skill", "source": "plugin"},
                {"name": "project-skill", "source": "project"},
            ],
            "commands": [
                {"name": "user-cmd", "source": "user"},
                {"name": "project-cmd", "source": "project"},
            ],
        }
        result = _project_scoped(features)
        assert len(result["skills"]) == 1
        assert result["skills"][0]["name"] == "project-skill"
        assert len(result["commands"]) == 1
        assert result["commands"][0]["name"] == "project-cmd"

    def test_drops_counts_non_list_key(self):
        _project_scoped = self._get_helper()
        features = {
            "skills": [{"name": "s", "source": "project"}],
            "counts": {"skills": 1},
        }
        result = _project_scoped(features)
        assert "counts" not in result

    def test_empty_list_when_no_project_items(self):
        _project_scoped = self._get_helper()
        features = {
            "skills": [
                {"name": "user-skill", "source": "user"},
            ],
        }
        result = _project_scoped(features)
        assert result["skills"] == []

    def test_handles_non_dict_items_gracefully(self):
        _project_scoped = self._get_helper()
        features = {
            "skills": ["not-a-dict", {"name": "p", "source": "project"}],
        }
        result = _project_scoped(features)
        # non-dict items are silently dropped
        assert result["skills"] == [{"name": "p", "source": "project"}]


# ---------------------------------------------------------------------------
# (e) harness totals — user items counted once even with multiple project repos
# ---------------------------------------------------------------------------

class TestHarnessTotals:
    """Test the totals logic directly without a full Flask app."""

    def _simulate_totals(self, user_features, project_features_list):
        """Replicate the harness() totals calculation."""
        from claudesavvy.web.routes.dashboard import _project_scoped

        def _count(features, key):
            items = features.get(key) or []
            return len(items) if isinstance(items, list) else 0

        all_features = [user_features] + [_project_scoped(pf) for pf in project_features_list]
        return {
            "skills": sum(_count(f, "skills") for f in all_features),
            "commands": sum(_count(f, "commands") for f in all_features),
        }

    def test_user_skills_counted_once_across_many_repos(self):
        # Simulate: 3 user-level skills returned for every repo (the bug scenario)
        user_skills = [{"name": f"u{i}", "source": "user"} for i in range(3)]
        user_features = {"skills": user_skills, "commands": []}

        # Two project repos both "return" the same 3 user skills (old bug)
        # plus one project skill each
        project_features_list = [
            {
                "skills": user_skills + [{"name": "proj-skill-1", "source": "project"}],
                "commands": [],
            },
            {
                "skills": user_skills + [{"name": "proj-skill-2", "source": "project"}],
                "commands": [],
            },
        ]

        totals = self._simulate_totals(user_features, project_features_list)
        # Should be 3 (user) + 1 + 1 (project) = 5, NOT 3 + 3*3 = 12
        assert totals["skills"] == 5

    def test_project_commands_counted_per_repo(self):
        user_cmds = [{"name": "uc", "source": "user"}]
        user_features = {"skills": [], "commands": user_cmds}

        project_features_list = [
            {"skills": [], "commands": user_cmds + [{"name": "pc1", "source": "project"}]},
            {"skills": [], "commands": user_cmds + [{"name": "pc2", "source": "project"}]},
        ]

        totals = self._simulate_totals(user_features, project_features_list)
        # 1 (user) + 1 + 1 (project) = 3
        assert totals["commands"] == 3

    def test_zero_project_items_gives_user_only_count(self):
        user_features = {
            "skills": [{"name": "s", "source": "user"}],
            "commands": [],
        }
        # 5 repos, each returning the same user skill but no project skills
        project_features_list = [
            {"skills": [{"name": "s", "source": "user"}], "commands": []}
            for _ in range(5)
        ]

        totals = self._simulate_totals(user_features, project_features_list)
        assert totals["skills"] == 1
