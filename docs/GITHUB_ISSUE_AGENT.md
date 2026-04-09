# GitHub Issue Agent

The GitHub Issue Agent is an automation tool that helps manage GitHub issues by automatically selecting and processing them based on priority.

## Features

- **Automatic Issue Fetching**: Retrieves all open issues from a GitHub repository
- **Smart Prioritization**: Ranks issues based on labels and age
  - Bugs and critical issues get highest priority
  - "Good first issue" labels get second priority
  - Enhancements and features get third priority
  - Older issues within each category are prioritized
- **Intelligent Selection**: Filters out assigned, duplicate, and invalid issues
- **Multiple Interfaces**: Available via CLI command, standalone script, or GitHub Actions workflow

## Priority System

Issues are prioritized in the following order:

1. **Highest Priority**: Issues with `bug` or `critical` labels
2. **High Priority**: Issues with `good first issue` label
3. **Medium Priority**: Issues with `enhancement` or `feature` labels
4. **Low Priority**: All other issues

Within each priority level, older issues are selected first.

### Filtered Labels

Issues with the following labels are automatically excluded:
- `wontfix`
- `duplicate`
- `invalid`

Issues that are already assigned to someone are also excluded.

## Installation

The GitHub Issue Agent is included with ClaudeSavvy:

```bash
pip install claudesavvy
```

Or install from source:

```bash
git clone https://github.com/allannapier/claudesavvy.git
cd claudesavvy
pip install -e .
```

## Prerequisites

- **GitHub CLI (`gh`)**: Must be installed and authenticated
  ```bash
  # Install gh CLI (macOS)
  brew install gh

  # Install gh CLI (Linux)
  # See: https://github.com/cli/cli/blob/trunk/docs/install_linux.md

  # Authenticate
  gh auth login
  ```

- **Python 3.9+**: Required for running the agent

## Usage

### CLI Command

The easiest way to use the agent is via the `claudesavvy` CLI:

```bash
# Run in dry-run mode to see which issue would be selected
claudesavvy issue-agent --repo-owner allannapier --repo-name claudesavvy --dry-run

# Run the agent (currently only selects, doesn't create PR yet)
claudesavvy issue-agent --repo-owner allannapier --repo-name claudesavvy

# Use a custom GitHub token
claudesavvy issue-agent \
  --repo-owner allannapier \
  --repo-name claudesavvy \
  --github-token ghp_your_token_here
```

### Standalone Script

You can also run the agent as a standalone Python script:

```bash
python scripts/run_issue_agent.py \
  --repo-owner allannapier \
  --repo-name claudesavvy \
  --dry-run
```

### GitHub Actions Workflow

A workflow is provided to run the agent automatically on a schedule:

```yaml
# .github/workflows/issue-agent.yml
name: GitHub Issue Agent

on:
  schedule:
    # Run daily at 9 AM UTC
    - cron: '0 9 * * *'
  workflow_dispatch:  # Allow manual triggering
```

To enable the workflow:
1. The workflow file is already in `.github/workflows/issue-agent.yml`
2. Make sure the repository has the necessary permissions (contents: write, issues: read, pull-requests: write)
3. The workflow will run automatically on schedule or can be triggered manually via Actions tab

## Python API

You can also use the agent programmatically in your own Python code:

```python
from claudesavvy.agents.github_issue_agent import GitHubIssueAgent

# Initialize the agent
agent = GitHubIssueAgent(
    repo_owner='allannapier',
    repo_name='claudesavvy',
    github_token='ghp_your_token'  # Optional, uses GITHUB_TOKEN env var
)

# Run the agent
result = agent.run()

# Check the result
if result['status'] == 'issue_selected':
    issue = result['issue']
    print(f"Selected issue #{issue['number']}: {issue['title']}")
elif result['status'] == 'no_issues':
    print("No open issues found")
elif result['status'] == 'no_available_issues':
    print("No available issues to work on")
```

## Configuration

### Environment Variables

- `GITHUB_TOKEN`: GitHub personal access token (required if not using `gh` CLI authentication)

### Options

- `--repo-owner`: GitHub repository owner (required)
- `--repo-name`: GitHub repository name (required)
- `--github-token`: GitHub token (optional, uses env var if not provided)
- `--dry-run`: Run without taking action, just show what would be selected

## Example Output

```
=== GitHub Issue Agent ===
Repository: allannapier/claudesavvy

✓ Selected issue to work on:

Issue Number  #42
Title         Add support for custom time ranges
Labels        enhancement, good first issue
URL           https://github.com/allannapier/claudesavvy/issues/42

Note: Automatic branch and PR creation is not yet implemented
You can manually create a branch and PR for this issue
```

## Future Enhancements

The following features are planned for future releases:

- [ ] Automatic branch creation for selected issues
- [ ] AI-powered PR generation using Claude Code
- [ ] Automatic PR submission
- [ ] Issue assignment before starting work
- [ ] Configurable priority rules via config file
- [ ] Support for multiple repositories
- [ ] Webhook integration for real-time processing

## Troubleshooting

### "Failed to fetch issues"

Make sure:
1. The `gh` CLI is installed and authenticated (`gh auth login`)
2. You have access to the repository
3. The repository owner and name are correct

### "No available issues to work on"

This means either:
- All issues are assigned to someone
- All issues have filtering labels (`wontfix`, `duplicate`, `invalid`)
- There are no open issues in the repository

### "Command 'gh' not found"

Install the GitHub CLI:
- macOS: `brew install gh`
- Linux: See [installation instructions](https://github.com/cli/cli/blob/trunk/docs/install_linux.md)
- Windows: `winget install --id GitHub.cli`

## Contributing

Contributions are welcome! To add features or improvements:

1. Fork the repository
2. Create a feature branch
3. Make your changes to `src/claudesavvy/agents/github_issue_agent.py`
4. Add tests if applicable
5. Submit a pull request

## License

This feature is part of ClaudeSavvy and is available under the MIT License.

## Support

- **Issues**: [GitHub Issues](https://github.com/allannapier/claudesavvy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/allannapier/claudesavvy/discussions)
