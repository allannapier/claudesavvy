# GitHub Code Review Agent

The GitHub Code Review Agent is an automation tool that performs automated code reviews on GitHub pull requests by analyzing code changes for security issues, code quality problems, and best practices.

## Features

- **Automatic PR Fetching**: Retrieves all open pull requests from a GitHub repository
- **Security Analysis**: Detects potential security vulnerabilities including:
  - Hardcoded secrets (passwords, API keys, tokens)
  - SQL injection vulnerabilities
  - Code injection risks (eval/exec usage)
  - Log injection vulnerabilities
- **Code Quality Checks**: Identifies code quality issues such as:
  - Debug logging statements
  - TODO/FIXME comments
  - Code smell patterns
- **Multiple Interfaces**: Available via CLI command, standalone script, or GitHub Actions workflow
- **Detailed Reports**: Provides categorized feedback with severity levels (error, warning, info)

## Installation

The GitHub Code Review Agent is included with ClaudeSavvy:

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
# List all open pull requests
claudesavvy code-review --repo-owner allannapier --repo-name claudesavvy

# Review a specific pull request
claudesavvy code-review --repo-owner allannapier --repo-name claudesavvy --pr-number 68

# Use a custom GitHub token
claudesavvy code-review \
  --repo-owner allannapier \
  --repo-name claudesavvy \
  --pr-number 68 \
  --github-token ghp_your_token_here
```

### Standalone Script

You can also run the agent as a standalone Python script:

```bash
python scripts/run_code_review_agent.py \
  --repo-owner allannapier \
  --repo-name claudesavvy \
  --pr-number 68
```

### GitHub Actions Workflow

A workflow is provided to automatically review pull requests:

```yaml
# .github/workflows/code-review-agent.yml
name: GitHub Code Review Agent

on:
  pull_request:
    types: [opened, synchronize, reopened]
  workflow_dispatch:
```

The workflow will:
1. Run automatically when PRs are opened, updated, or reopened
2. Can be triggered manually via Actions tab
3. Analyze the PR for security and quality issues
4. Display results in the workflow logs

To enable the workflow:
1. The workflow file is already in `.github/workflows/code-review-agent.yml`
2. Make sure the repository has the necessary permissions (contents: read, pull-requests: write)
3. The workflow will run automatically on PR events

## Python API

You can also use the agent programmatically in your own Python code:

```python
from claudesavvy.agents.github_code_review_agent import GitHubCodeReviewAgent

# Initialize the agent
agent = GitHubCodeReviewAgent(
    repo_owner='allannapier',
    repo_name='claudesavvy',
    github_token='ghp_your_token'  # Optional, uses GITHUB_TOKEN env var
)

# List all open PRs
result = agent.run()
if result['status'] == 'prs_listed':
    for pr in result['prs']:
        print(f"PR #{pr['number']}: {pr['title']}")

# Review a specific PR
result = agent.run(pr_number=68)
if result['status'] == 'review_complete':
    review = result['review']
    print(f"Found {review['total_comments']} issues")
    print(f"Errors: {review['errors']}, Warnings: {review['warnings']}")
```

## Configuration

### Environment Variables

- `GITHUB_TOKEN`: GitHub personal access token (required if not using `gh` CLI authentication)

### Options

- `--repo-owner`: GitHub repository owner (required)
- `--repo-name`: GitHub repository name (required)
- `--pr-number`: Pull request number to review (optional, lists all PRs if not provided)
- `--github-token`: GitHub token (optional, uses env var if not provided)

## Security Checks

The agent performs the following security checks:

### Hardcoded Secrets
Detects patterns like:
- `password =`
- `api_key =`
- `secret =`
- `token =`

**Severity**: Error

### SQL Injection
Detects potentially unsafe SQL patterns:
- `execute(f"`
- `execute("SELECT`
- String formatting in SQL queries

**Severity**: Error

### Code Injection
Detects dangerous eval/exec usage:
- `eval(`
- `exec(`

**Severity**: Warning

### Log Injection
Detects potential log injection vulnerabilities:
- `logger.info(f`
- `print(f`
- `console.log(`

**Severity**: Warning (per CLAUDE.md project guidelines)

## Code Quality Checks

### Debug Logging
Detects debug statements that should be removed:
- `console.log(`
- `print("DEBUG`
- `logger.debug(`

**Severity**: Info

### TODO/FIXME Comments
Identifies unresolved TODOs:
- `# TODO`
- `// TODO`
- `# FIXME`
- `// FIXME`

**Severity**: Info

## Example Output

### Listing PRs

```
=== GitHub Code Review Agent ===
Repository: allannapier/claudesavvy

✓ Open pull requests:

PR #  Title                           Author      Files  +/-
#68   Add GitHub Issue Agent          Claude      10     +1069/-0
#69   Update dependencies             allannapier 2      +15/-5

Use --pr-number to review a specific pull request
```

### Reviewing a PR

```
=== GitHub Code Review Agent ===
Repository: allannapier/claudesavvy

✓ Review complete for PR #68:

Title         Add GitHub Issue Agent for automated issue processing
Author        Claude
Changed Files 10
Changes       +1069 / -0
URL           https://github.com/allannapier/claudesavvy/pull/68

Review Summary:
  Total Comments: 3
  Errors: 1
  Warnings: 1
  Info: 1

Errors:
  ✗ src/claudesavvy/agents/github_issue_agent.py:142
    Potential hardcoded secret detected: `token =`

Warnings:
  ⚠ src/claudesavvy/cli.py:335
    Potential log injection - avoid logging user input directly: `logger.info(f`

Info:
  ℹ scripts/run_issue_agent.py:45
    TODO/FIXME comment found - consider creating an issue
```

## Integration with CI/CD

You can integrate the code review agent into your CI/CD pipeline:

### GitHub Actions Example

```yaml
name: Code Review

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.9'
      - name: Install ClaudeSavvy
        run: pip install claudesavvy
      - name: Run Code Review
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          claudesavvy code-review \
            --repo-owner ${{ github.repository_owner }} \
            --repo-name ${{ github.event.repository.name }} \
            --pr-number ${{ github.event.pull_request.number }}
```

## Future Enhancements

The following features are planned for future releases:

- [ ] Automatic posting of review comments directly to PRs
- [ ] AI-powered code review using Claude Code
- [ ] Custom rule configuration via config file
- [ ] Support for multiple languages (currently focused on Python/JavaScript)
- [ ] Integration with code quality tools (pylint, eslint, etc.)
- [ ] Performance analysis
- [ ] Test coverage analysis
- [ ] Dependency vulnerability scanning

## Troubleshooting

### "Failed to fetch pull requests"

Make sure:
1. The `gh` CLI is installed and authenticated (`gh auth login`)
2. You have access to the repository
3. The repository owner and name are correct

### "Pull request not found"

This means either:
- The PR number doesn't exist
- The PR has been closed or merged
- You don't have access to view the PR

### "Command 'gh' not found"

Install the GitHub CLI:
- macOS: `brew install gh`
- Linux: See [installation instructions](https://github.com/cli/cli/blob/trunk/docs/install_linux.md)
- Windows: `winget install --id GitHub.cli`

### Review finds false positives

The automated review uses pattern matching and may produce false positives. Use your judgment when addressing feedback. Future versions will include:
- Configurable rules
- AI-powered analysis for better accuracy
- Ability to suppress specific warnings

## Contributing

Contributions are welcome! To add features or improvements:

1. Fork the repository
2. Create a feature branch
3. Make your changes to `src/claudesavvy/agents/github_code_review_agent.py`
4. Add tests if applicable
5. Submit a pull request

## License

This feature is part of ClaudeSavvy and is available under the MIT License.

## Support

- **Issues**: [GitHub Issues](https://github.com/allannapier/claudesavvy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/allannapier/claudesavvy/discussions)
