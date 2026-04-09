#!/usr/bin/env python3
"""
Standalone script to run the GitHub issue agent.

This script can be run directly without installing the package:
    python scripts/run_issue_agent.py --repo-owner allannapier --repo-name claudesavvy

Requirements:
    - gh CLI must be installed and authenticated
    - Python 3.9 or higher
"""

import argparse
import sys
from pathlib import Path

# Add src directory to path to import claudesavvy modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from claudesavvy.agents.github_issue_agent import GitHubIssueAgent


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Run the GitHub issue agent to process the next issue'
    )
    parser.add_argument(
        '--repo-owner',
        required=True,
        help='GitHub repository owner'
    )
    parser.add_argument(
        '--repo-name',
        required=True,
        help='GitHub repository name'
    )
    parser.add_argument(
        '--github-token',
        help='GitHub personal access token (uses GITHUB_TOKEN env var if not provided)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show which issue would be selected without taking action'
    )

    args = parser.parse_args()

    # Initialize and run agent
    print(f"\n=== GitHub Issue Agent ===")
    print(f"Repository: {args.repo_owner}/{args.repo_name}\n")

    try:
        agent = GitHubIssueAgent(args.repo_owner, args.repo_name, args.github_token)

        if args.dry_run:
            print("Running in dry-run mode...\n")

        result = agent.run()

        # Display results
        status = result['status']

        if status == 'no_issues':
            print("✓ No open issues found in the repository\n")
        elif status == 'no_available_issues':
            print("✓ No available issues to work on")
            print("All issues are either assigned or filtered out\n")
        elif status == 'issue_selected':
            issue = result['issue']
            print("✓ Selected issue to work on:\n")
            print(f"  Issue Number: #{issue['number']}")
            print(f"  Title: {issue['title']}")
            print(f"  Labels: {', '.join(issue['labels']) if issue['labels'] else 'None'}")
            print(f"  URL: {issue['url']}\n")

            if args.dry_run:
                print("Dry-run mode: No branch or PR will be created\n")
            else:
                print("Note: Automatic branch and PR creation is not yet implemented")
                print("You can manually create a branch and PR for this issue\n")
        else:
            print(f"✗ Unknown status: {status}\n")
            sys.exit(1)

    except RuntimeError as e:
        print(f"\n✗ Error: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
