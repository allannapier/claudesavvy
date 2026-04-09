"""GitHub Issue Agent for automatically processing issues and creating PRs."""

import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any
import json


@dataclass
class Issue:
    """Represents a GitHub issue."""
    number: int
    title: str
    body: Optional[str]
    labels: List[str]
    state: str
    created_at: str
    updated_at: str
    assignees: List[str]
    url: str

    @classmethod
    def from_github_api(cls, issue_data: Dict[str, Any]) -> 'Issue':
        """Create Issue from GitHub API response."""
        return cls(
            number=issue_data['number'],
            title=issue_data['title'],
            body=issue_data.get('body', ''),
            labels=[label.get('name', '') for label in issue_data.get('labels', [])],
            state=issue_data['state'],
            created_at=issue_data['createdAt'],
            updated_at=issue_data['updatedAt'],
            assignees=[assignee.get('login', '') for assignee in issue_data.get('assignees', {}).get('nodes', [])],
            url=issue_data.get('url', ''),
        )


class GitHubIssueAgent:
    """Agent that processes GitHub issues and creates PRs to fix them."""

    def __init__(self, repo_owner: str, repo_name: str, github_token: Optional[str] = None):
        """Initialize the GitHub issue agent.

        Args:
            repo_owner: GitHub repository owner
            repo_name: GitHub repository name
            github_token: GitHub personal access token (optional, uses env var if not provided)
        """
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')

    def fetch_open_issues(self) -> List[Issue]:
        """Fetch all open issues from the GitHub repository.

        Returns:
            List of Issue objects
        """
        # Use gh CLI to fetch issues
        try:
            cmd = [
                'gh', 'issue', 'list',
                '--repo', f'{self.repo_owner}/{self.repo_name}',
                '--state', 'open',
                '--json', 'number,title,body,labels,state,createdAt,updatedAt,assignees,url',
                '--limit', '100'
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            issues_data = json.loads(result.stdout)
            return [Issue.from_github_api(issue) for issue in issues_data]

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to fetch issues: {e.stderr}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse issue data: {e}")

    def prioritize_issues(self, issues: List[Issue]) -> List[Issue]:
        """Prioritize issues based on labels, age, and other criteria.

        Priority order:
        1. Issues with 'bug' label (highest priority)
        2. Issues with 'good first issue' label
        3. Issues with 'enhancement' label
        4. Other issues

        Within each category, older issues are prioritized.

        Args:
            issues: List of Issue objects

        Returns:
            Sorted list of issues by priority
        """
        def get_priority_score(issue: Issue) -> tuple:
            """Calculate priority score for an issue."""
            # Priority categories (lower number = higher priority)
            if 'bug' in issue.labels or 'critical' in issue.labels:
                category = 0
            elif 'good first issue' in issue.labels:
                category = 1
            elif 'enhancement' in issue.labels or 'feature' in issue.labels:
                category = 2
            else:
                category = 3

            # Parse created_at for age-based sorting
            try:
                created_date = datetime.fromisoformat(issue.created_at.replace('Z', '+00:00'))
                age_score = created_date.timestamp()
            except (ValueError, AttributeError):
                age_score = 0

            # Return tuple for sorting (category first, then age - older is higher priority)
            return (category, age_score)

        return sorted(issues, key=get_priority_score)

    def select_next_issue(self, issues: List[Issue]) -> Optional[Issue]:
        """Select the next issue to work on.

        Filters out issues that are already assigned or have certain labels,
        then returns the highest priority issue.

        Args:
            issues: List of Issue objects

        Returns:
            The next issue to work on, or None if no suitable issue found
        """
        # Filter out assigned issues and issues with 'wontfix' or 'duplicate' labels
        available_issues = [
            issue for issue in issues
            if not issue.assignees
            and 'wontfix' not in issue.labels
            and 'duplicate' not in issue.labels
            and 'invalid' not in issue.labels
        ]

        if not available_issues:
            return None

        # Prioritize and return the top issue
        prioritized = self.prioritize_issues(available_issues)
        return prioritized[0] if prioritized else None

    def create_branch_for_issue(self, issue: Issue) -> str:
        """Create a new git branch for working on the issue.

        Args:
            issue: The issue to create a branch for

        Returns:
            The branch name created
        """
        # Create branch name from issue number and title
        # Format: issue-{number}-{sanitized-title}
        sanitized_title = issue.title.lower()
        # Remove special characters and replace spaces with hyphens
        sanitized_title = ''.join(c if c.isalnum() or c.isspace() else '' for c in sanitized_title)
        sanitized_title = '-'.join(sanitized_title.split())[:50]  # Limit length

        branch_name = f"issue-{issue.number}-{sanitized_title}"

        # Create and checkout the branch
        try:
            subprocess.run(['git', 'checkout', '-b', branch_name], check=True)
            return branch_name
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create branch: {e}")

    def create_pr_for_issue(self, issue: Issue, branch_name: str, commit_message: str) -> str:
        """Create a pull request for the issue.

        Args:
            issue: The issue this PR addresses
            branch_name: The branch containing the changes
            commit_message: Summary of changes made

        Returns:
            The URL of the created PR
        """
        # Push the branch
        try:
            subprocess.run(['git', 'push', '-u', 'origin', branch_name], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to push branch: {e}")

        # Create PR using gh CLI
        pr_title = f"Fix #{issue.number}: {issue.title}"
        pr_body = f"""## Summary
This PR addresses issue #{issue.number}.

## Changes
{commit_message}

## Related Issue
Fixes #{issue.number}
"""

        try:
            cmd = [
                'gh', 'pr', 'create',
                '--title', pr_title,
                '--body', pr_body,
                '--repo', f'{self.repo_owner}/{self.repo_name}'
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            return result.stdout.strip()

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create PR: {e.stderr}")

    def run(self) -> Dict[str, Any]:
        """Run the agent to process the next issue.

        Returns:
            Dictionary with agent run results including issue processed and PR created
        """
        # Fetch open issues
        issues = self.fetch_open_issues()

        if not issues:
            return {
                'status': 'no_issues',
                'message': 'No open issues found'
            }

        # Select next issue
        next_issue = self.select_next_issue(issues)

        if not next_issue:
            return {
                'status': 'no_available_issues',
                'message': 'No available issues to work on (all are assigned or filtered out)'
            }

        return {
            'status': 'issue_selected',
            'issue': {
                'number': next_issue.number,
                'title': next_issue.title,
                'url': next_issue.url,
                'labels': next_issue.labels
            },
            'message': f'Selected issue #{next_issue.number}: {next_issue.title}'
        }
