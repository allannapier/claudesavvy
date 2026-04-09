"""GitHub Code Review Agent for automated PR reviews."""

import os
import subprocess
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import json


@dataclass
class PullRequest:
    """Represents a GitHub pull request."""
    number: int
    title: str
    body: Optional[str]
    state: str
    author: str
    base_branch: str
    head_branch: str
    created_at: str
    updated_at: str
    url: str
    additions: int
    deletions: int
    changed_files: int

    @classmethod
    def from_github_api(cls, pr_data: Dict[str, Any]) -> 'PullRequest':
        """Create PullRequest from GitHub API response."""
        return cls(
            number=pr_data['number'],
            title=pr_data['title'],
            body=pr_data.get('body', ''),
            state=pr_data['state'],
            author=pr_data.get('author', {}).get('login', 'unknown'),
            base_branch=pr_data.get('baseRefName', 'main'),
            head_branch=pr_data.get('headRefName', ''),
            created_at=pr_data['createdAt'],
            updated_at=pr_data['updatedAt'],
            url=pr_data.get('url', ''),
            additions=pr_data.get('additions', 0),
            deletions=pr_data.get('deletions', 0),
            changed_files=pr_data.get('changedFiles', 0),
        )


@dataclass
class ReviewComment:
    """Represents a code review comment."""
    file_path: str
    line_number: int
    comment: str
    severity: str  # 'error', 'warning', 'info'


class GitHubCodeReviewAgent:
    """Agent that performs automated code reviews on GitHub PRs."""

    def __init__(self, repo_owner: str, repo_name: str, github_token: Optional[str] = None):
        """Initialize the GitHub code review agent.

        Args:
            repo_owner: GitHub repository owner
            repo_name: GitHub repository name
            github_token: GitHub personal access token (optional, uses env var if not provided)
        """
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')

    def fetch_open_prs(self) -> List[PullRequest]:
        """Fetch all open pull requests from the GitHub repository.

        Returns:
            List of PullRequest objects
        """
        try:
            cmd = [
                'gh', 'pr', 'list',
                '--repo', f'{self.repo_owner}/{self.repo_name}',
                '--state', 'open',
                '--json', 'number,title,body,state,author,baseRefName,headRefName,createdAt,updatedAt,url,additions,deletions,changedFiles',
                '--limit', '100'
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            prs_data = json.loads(result.stdout)
            return [PullRequest.from_github_api(pr) for pr in prs_data]

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to fetch pull requests: {e.stderr}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse pull request data: {e}")

    def get_pr_diff(self, pr_number: int) -> str:
        """Get the diff for a pull request.

        Args:
            pr_number: Pull request number

        Returns:
            The diff as a string
        """
        try:
            cmd = [
                'gh', 'pr', 'diff',
                str(pr_number),
                '--repo', f'{self.repo_owner}/{self.repo_name}'
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            return result.stdout

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to get PR diff: {e.stderr}")

    def get_pr_files(self, pr_number: int) -> List[str]:
        """Get list of files changed in a pull request.

        Args:
            pr_number: Pull request number

        Returns:
            List of file paths
        """
        try:
            cmd = [
                'gh', 'pr', 'view',
                str(pr_number),
                '--repo', f'{self.repo_owner}/{self.repo_name}',
                '--json', 'files'
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            data = json.loads(result.stdout)
            files = data.get('files', [])
            return [f.get('path', '') for f in files]

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to get PR files: {e.stderr}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse PR files data: {e}")

    def analyze_security_issues(self, diff: str) -> List[ReviewComment]:
        """Analyze diff for potential security issues.

        Args:
            diff: The diff content

        Returns:
            List of ReviewComment objects with security findings
        """
        comments = []

        # Basic security checks
        security_patterns = {
            'hardcoded_secret': {
                'patterns': ['password =', 'api_key =', 'secret =', 'token ='],
                'message': 'Potential hardcoded secret detected',
                'severity': 'error'
            },
            'sql_injection': {
                'patterns': ['execute(f"', 'execute("SELECT', '.format('],
                'message': 'Potential SQL injection vulnerability',
                'severity': 'error'
            },
            'eval_usage': {
                'patterns': ['eval(', 'exec('],
                'message': 'Use of eval/exec detected - potential code injection risk',
                'severity': 'warning'
            },
            'logging_user_input': {
                'patterns': ['logger.info(f', 'print(f', 'console.log('],
                'message': 'Potential log injection - avoid logging user input directly',
                'severity': 'warning'
            }
        }

        lines = diff.split('\n')
        current_file = None
        line_number = 0

        for line in lines:
            if line.startswith('diff --git'):
                parts = line.split()
                if len(parts) >= 3:
                    current_file = parts[2].lstrip('a/')
                line_number = 0
            elif line.startswith('@@'):
                # Extract line number from hunk header
                try:
                    parts = line.split('+')[1].split(',')[0]
                    line_number = int(parts)
                except (IndexError, ValueError):
                    line_number = 0
            elif line.startswith('+') and not line.startswith('+++'):
                line_number += 1
                line_content = line[1:].strip()

                for check_name, check_info in security_patterns.items():
                    for pattern in check_info['patterns']:
                        if pattern in line_content:
                            comments.append(ReviewComment(
                                file_path=current_file or 'unknown',
                                line_number=line_number,
                                comment=f"{check_info['message']}: `{pattern}`",
                                severity=check_info['severity']
                            ))

        return comments

    def analyze_code_quality(self, diff: str) -> List[ReviewComment]:
        """Analyze diff for code quality issues.

        Args:
            diff: The diff content

        Returns:
            List of ReviewComment objects with quality findings
        """
        comments = []

        quality_patterns = {
            'debug_logging': {
                'patterns': ['console.log(', 'print("DEBUG', 'logger.debug('],
                'message': 'Debug logging statement found - should be removed before merging',
                'severity': 'info'
            },
            'todo_comment': {
                'patterns': ['# TODO', '// TODO', '# FIXME', '// FIXME'],
                'message': 'TODO/FIXME comment found - consider creating an issue',
                'severity': 'info'
            },
            'long_function': {
                'patterns': [],  # Handled separately
                'message': 'Function appears to be very long - consider breaking it down',
                'severity': 'info'
            }
        }

        lines = diff.split('\n')
        current_file = None
        line_number = 0

        for line in lines:
            if line.startswith('diff --git'):
                parts = line.split()
                if len(parts) >= 3:
                    current_file = parts[2].lstrip('a/')
                line_number = 0
            elif line.startswith('@@'):
                try:
                    parts = line.split('+')[1].split(',')[0]
                    line_number = int(parts)
                except (IndexError, ValueError):
                    line_number = 0
            elif line.startswith('+') and not line.startswith('+++'):
                line_number += 1
                line_content = line[1:].strip()

                for check_name, check_info in quality_patterns.items():
                    if check_name == 'long_function':
                        continue  # Skip pattern-based check for this

                    for pattern in check_info['patterns']:
                        if pattern in line_content:
                            comments.append(ReviewComment(
                                file_path=current_file or 'unknown',
                                line_number=line_number,
                                comment=check_info['message'],
                                severity=check_info['severity']
                            ))

        return comments

    def review_pr(self, pr_number: int) -> Dict[str, Any]:
        """Perform automated code review on a pull request.

        Args:
            pr_number: Pull request number to review

        Returns:
            Dictionary with review results
        """
        # Get PR details
        prs = self.fetch_open_prs()
        pr = next((p for p in prs if p.number == pr_number), None)

        if not pr:
            return {
                'status': 'pr_not_found',
                'message': f'Pull request #{pr_number} not found'
            }

        # Get diff
        diff = self.get_pr_diff(pr_number)

        # Analyze for issues
        security_comments = self.analyze_security_issues(diff)
        quality_comments = self.analyze_code_quality(diff)

        all_comments = security_comments + quality_comments

        # Categorize by severity
        errors = [c for c in all_comments if c.severity == 'error']
        warnings = [c for c in all_comments if c.severity == 'warning']
        info = [c for c in all_comments if c.severity == 'info']

        return {
            'status': 'review_complete',
            'pr': {
                'number': pr.number,
                'title': pr.title,
                'url': pr.url,
                'author': pr.author,
                'changed_files': pr.changed_files,
                'additions': pr.additions,
                'deletions': pr.deletions
            },
            'review': {
                'total_comments': len(all_comments),
                'errors': len(errors),
                'warnings': len(warnings),
                'info': len(info),
                'comments': [
                    {
                        'file': c.file_path,
                        'line': c.line_number,
                        'comment': c.comment,
                        'severity': c.severity
                    }
                    for c in all_comments
                ]
            }
        }

    def run(self, pr_number: Optional[int] = None) -> Dict[str, Any]:
        """Run the code review agent.

        Args:
            pr_number: Optional specific PR number to review. If not provided,
                      will list all open PRs

        Returns:
            Dictionary with agent run results
        """
        if pr_number:
            return self.review_pr(pr_number)

        # List all open PRs
        prs = self.fetch_open_prs()

        if not prs:
            return {
                'status': 'no_prs',
                'message': 'No open pull requests found'
            }

        return {
            'status': 'prs_listed',
            'prs': [
                {
                    'number': pr.number,
                    'title': pr.title,
                    'author': pr.author,
                    'url': pr.url,
                    'changed_files': pr.changed_files,
                    'additions': pr.additions,
                    'deletions': pr.deletions
                }
                for pr in prs
            ]
        }
