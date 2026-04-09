#!/usr/bin/env python3
"""
Standalone script to run the GitHub code review agent.

This script can be run directly without installing the package:
    python scripts/run_code_review_agent.py --repo-owner allannapier --repo-name claudesavvy

Requirements:
    - gh CLI must be installed and authenticated
    - Python 3.9 or higher
"""

import argparse
import sys
from pathlib import Path

# Add src directory to path to import claudesavvy modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from claudesavvy.agents.github_code_review_agent import GitHubCodeReviewAgent


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Run the GitHub code review agent to analyze pull requests'
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
        '--pr-number',
        type=int,
        help='Pull request number to review (if not provided, lists all open PRs)'
    )
    parser.add_argument(
        '--github-token',
        help='GitHub personal access token (uses GITHUB_TOKEN env var if not provided)'
    )

    args = parser.parse_args()

    # Initialize and run agent
    print(f"\n=== GitHub Code Review Agent ===")
    print(f"Repository: {args.repo_owner}/{args.repo_name}\n")

    try:
        agent = GitHubCodeReviewAgent(args.repo_owner, args.repo_name, args.github_token)

        result = agent.run(args.pr_number)

        # Display results
        status = result['status']

        if status == 'no_prs':
            print("✓ No open pull requests found in the repository\n")
        elif status == 'pr_not_found':
            print(f"✗ Pull request #{args.pr_number} not found\n")
            sys.exit(1)
        elif status == 'prs_listed':
            print("✓ Open pull requests:\n")
            for pr in result['prs']:
                print(f"  #{pr['number']}: {pr['title']}")
                print(f"    Author: {pr['author']}")
                print(f"    Files: {pr['changed_files']} (+{pr['additions']}/-{pr['deletions']})")
                print(f"    URL: {pr['url']}\n")
            print("Use --pr-number to review a specific pull request\n")

        elif status == 'review_complete':
            pr = result['pr']
            review = result['review']

            print(f"✓ Review complete for PR #{pr['number']}:\n")
            print(f"  Title: {pr['title']}")
            print(f"  Author: {pr['author']}")
            print(f"  Changed Files: {pr['changed_files']}")
            print(f"  Changes: +{pr['additions']} / -{pr['deletions']}")
            print(f"  URL: {pr['url']}\n")

            print("Review Summary:")
            print(f"  Total Comments: {review['total_comments']}")
            print(f"  Errors: {review['errors']}")
            print(f"  Warnings: {review['warnings']}")
            print(f"  Info: {review['info']}\n")

            if review['total_comments'] == 0:
                print("✓ No issues found!\n")
            else:
                # Group comments by severity
                errors = [c for c in review['comments'] if c['severity'] == 'error']
                warnings = [c for c in review['comments'] if c['severity'] == 'warning']
                info = [c for c in review['comments'] if c['severity'] == 'info']

                # Display errors
                if errors:
                    print("Errors:")
                    for comment in errors:
                        print(f"  ✗ {comment['file']}:{comment['line']}")
                        print(f"    {comment['comment']}\n")

                # Display warnings
                if warnings:
                    print("Warnings:")
                    for comment in warnings:
                        print(f"  ⚠ {comment['file']}:{comment['line']}")
                        print(f"    {comment['comment']}\n")

                # Display info
                if info:
                    print("Info:")
                    for comment in info:
                        print(f"  ℹ {comment['file']}:{comment['line']}")
                        print(f"    {comment['comment']}\n")

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
