"""Tests for the GitHub Issue Agent."""

import pytest
from datetime import datetime
from claudesavvy.agents.github_issue_agent import Issue, GitHubIssueAgent


def test_issue_from_github_api():
    """Test creating Issue from GitHub API data."""
    api_data = {
        'number': 42,
        'title': 'Test issue',
        'body': 'This is a test issue',
        'labels': [{'name': 'bug'}, {'name': 'critical'}],
        'state': 'open',
        'createdAt': '2024-01-01T00:00:00Z',
        'updatedAt': '2024-01-02T00:00:00Z',
        'assignees': {'nodes': [{'login': 'user1'}]},
        'url': 'https://github.com/test/test/issues/42'
    }

    issue = Issue.from_github_api(api_data)

    assert issue.number == 42
    assert issue.title == 'Test issue'
    assert issue.body == 'This is a test issue'
    assert issue.labels == ['bug', 'critical']
    assert issue.state == 'open'
    assert issue.assignees == ['user1']
    assert issue.url == 'https://github.com/test/test/issues/42'


def test_prioritize_issues():
    """Test issue prioritization logic."""
    agent = GitHubIssueAgent('test', 'test')

    # Create test issues
    bug_issue = Issue(
        number=1, title='Bug', body='', labels=['bug'],
        state='open', created_at='2024-01-01T00:00:00Z',
        updated_at='2024-01-01T00:00:00Z', assignees=[], url=''
    )

    enhancement_issue = Issue(
        number=2, title='Enhancement', body='', labels=['enhancement'],
        state='open', created_at='2024-01-02T00:00:00Z',
        updated_at='2024-01-02T00:00:00Z', assignees=[], url=''
    )

    good_first_issue = Issue(
        number=3, title='Good first issue', body='', labels=['good first issue'],
        state='open', created_at='2024-01-03T00:00:00Z',
        updated_at='2024-01-03T00:00:00Z', assignees=[], url=''
    )

    issues = [enhancement_issue, good_first_issue, bug_issue]

    # Prioritize
    prioritized = agent.prioritize_issues(issues)

    # Bug should be first, then good first issue, then enhancement
    assert prioritized[0].number == 1  # bug
    assert prioritized[1].number == 3  # good first issue
    assert prioritized[2].number == 2  # enhancement


def test_select_next_issue_filters_assigned():
    """Test that assigned issues are filtered out."""
    agent = GitHubIssueAgent('test', 'test')

    assigned_issue = Issue(
        number=1, title='Assigned', body='', labels=[],
        state='open', created_at='2024-01-01T00:00:00Z',
        updated_at='2024-01-01T00:00:00Z',
        assignees=['user1'], url=''
    )

    unassigned_issue = Issue(
        number=2, title='Unassigned', body='', labels=[],
        state='open', created_at='2024-01-02T00:00:00Z',
        updated_at='2024-01-02T00:00:00Z',
        assignees=[], url=''
    )

    issues = [assigned_issue, unassigned_issue]

    selected = agent.select_next_issue(issues)

    # Should select the unassigned issue
    assert selected is not None
    assert selected.number == 2


def test_select_next_issue_filters_wontfix():
    """Test that issues with wontfix label are filtered out."""
    agent = GitHubIssueAgent('test', 'test')

    wontfix_issue = Issue(
        number=1, title='Wontfix', body='', labels=['wontfix'],
        state='open', created_at='2024-01-01T00:00:00Z',
        updated_at='2024-01-01T00:00:00Z',
        assignees=[], url=''
    )

    valid_issue = Issue(
        number=2, title='Valid', body='', labels=['bug'],
        state='open', created_at='2024-01-02T00:00:00Z',
        updated_at='2024-01-02T00:00:00Z',
        assignees=[], url=''
    )

    issues = [wontfix_issue, valid_issue]

    selected = agent.select_next_issue(issues)

    # Should select the valid issue
    assert selected is not None
    assert selected.number == 2


def test_select_next_issue_returns_none_when_no_valid_issues():
    """Test that None is returned when no valid issues exist."""
    agent = GitHubIssueAgent('test', 'test')

    assigned_issue = Issue(
        number=1, title='Assigned', body='', labels=[],
        state='open', created_at='2024-01-01T00:00:00Z',
        updated_at='2024-01-01T00:00:00Z',
        assignees=['user1'], url=''
    )

    wontfix_issue = Issue(
        number=2, title='Wontfix', body='', labels=['wontfix'],
        state='open', created_at='2024-01-02T00:00:00Z',
        updated_at='2024-01-02T00:00:00Z',
        assignees=[], url=''
    )

    issues = [assigned_issue, wontfix_issue]

    selected = agent.select_next_issue(issues)

    # Should return None
    assert selected is None
