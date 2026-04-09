"""Main CLI interface for claudesavvy."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .utils.paths import get_claude_paths, ClaudeDataPaths
from .parsers.sessions import SessionParser
from .utils.time_filter import TimeFilter
from .agents.github_issue_agent import GitHubIssueAgent
from .agents.github_code_review_agent import GitHubCodeReviewAgent


def parse_time_filter(period: str) -> TimeFilter:
    """Parse time period string into TimeFilter."""
    now = datetime.now()
    
    if period == '1h':
        start = now - timedelta(hours=1)
    elif period == '24h':
        start = now - timedelta(hours=24)
    elif period == '7d':
        start = now - timedelta(days=7)
    elif period == '30d':
        start = now - timedelta(days=30)
    elif period == '90d':
        start = now - timedelta(days=90)
    else:
        # Default to 24h
        start = now - timedelta(hours=24)
    
    return TimeFilter(start_time=start, end_time=now)


def calculate_cost_from_tokens(total_tokens: int, tool_output_tokens: int = 0) -> float:
    """Calculate approximate cost from token count.
    
    Uses rough pricing: $3 per million input tokens, $15 per million output tokens
    Average ratio: ~70% input, 30% output
    Tool outputs are counted as input tokens.
    """
    # Rough average: $10 per million tokens (blended rate) for LLM tokens
    # Tool outputs are typically smaller but still count
    llm_tokens = total_tokens - tool_output_tokens
    return ((llm_tokens / 1_000_000) * 10.0) + ((tool_output_tokens / 1_000_000) * 10.0)


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """ClaudeSavvy - Usage tracking for Claude Code."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(serve)


@cli.command(name='serve')
@click.option(
    '--port',
    type=int,
    default=5000,
    help='Port to run web server on (default: 5000)'
)
@click.option(
    '--host',
    type=str,
    default='127.0.0.1',
    help='Host to bind to (default: 127.0.0.1)'
)
@click.option(
    '--debug',
    is_flag=True,
    help='Run in debug mode with auto-reload'
)
@click.option(
    '--claude-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help='Custom Claude data directory (default: ~/.claude/)'
)
def serve(port, host, debug, claude_dir):
    """
    Start the ClaudeSavvy web dashboard.

    Visualize your Claude Code usage metrics including sessions, token consumption,
    costs, projects, file operations, and MCP integrations through an intuitive
    web dashboard.

    Access the dashboard at http://localhost:5000 after starting the server.

    Examples:

      # Start server on default port 5000 (localhost-only, secure)
      $ claudesavvy serve

      # Start on custom port
      $ claudesavvy serve --port 8080

      # Make accessible from network (use with caution)
      $ claudesavvy serve --host 0.0.0.0

      # Enable debug mode with auto-reload
      $ claudesavvy serve --debug

      # Use custom Claude data directory
      $ claudesavvy serve --claude-dir /path/to/claude/data
    """
    console = Console()

    try:
        # Lazy import Flask to reduce startup time
        from .web.app import create_app

        # Get Claude data paths
        if claude_dir:
            paths = ClaudeDataPaths(Path(claude_dir))
        else:
            paths = get_claude_paths()

        # Create Flask app with paths
        app = create_app(paths)

        # Display startup information
        console.print("\n[cyan]═══ ClaudeSavvy Web Server ═══[/cyan]")
        console.print("[dim]Starting web interface...[/dim]\n")
        console.print(f"[green]✓ Server running at:[/green] [bold]http://{host}:{port}[/bold]")

        if debug:
            console.print("[yellow]⚠ Debug mode:[/yellow] [bold]Enabled[/bold] (auto-reload on code changes)")

        console.print(f"[dim]📊 Data source:[/dim] [dim]{paths.base_dir}[/dim]")
        console.print("\n[dim]Press Ctrl+C to stop the server[/dim]\n")

        # Run Flask app
        app.run(host=host, port=port, debug=debug)

    except FileNotFoundError as e:
        console.print(f"\n[red]✗ Error:[/red] {e}")
        console.print("\n[yellow]Make sure Claude Code has been used at least once.[/yellow]")
        console.print("[dim]ClaudeSavvy reads data from ~/.claude/ directory[/dim]\n")
        sys.exit(1)

    except ImportError as e:
        console.print(f"\n[red]✗ Import Error:[/red] {e}")
        console.print("\n[yellow]Flask is required for the web interface.[/yellow]")
        console.print("[dim]Install with: pip install flask[/dim]\n")
        sys.exit(1)

    except Exception as e:
        console.print(f"\n[red]✗ Unexpected error:[/red] {e}")
        console.print("\n[dim]If this issue persists, please report it at:")
        console.print("https://github.com/allannapier/claudesavvy/issues[/dim]\n")
        sys.exit(1)


@cli.command(name='stats')
@click.argument('period', default='24h')
@click.option(
    '--format',
    type=click.Choice(['json', 'table', 'text']),
    default='table',
    help='Output format (default: table)'
)
@click.option(
    '--claude-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help='Custom Claude data directory (default: ~/.claude/)'
)
def stats(period, format, claude_dir):
    """
    Show usage statistics per project for the specified time period.

    PERIOD can be: 1h, 24h, 7d, 30d, 90d

    Examples:

      # Show stats for last 24 hours (default)
      $ claudesavvy stats

      # Show stats for last week in JSON format
      $ claudesavvy stats 7d --format json

      # Show stats for last month
      $ claudesavvy stats 30d
    """
    console = Console()
    
    try:
        # Get Claude data paths
        if claude_dir:
            paths = ClaudeDataPaths(Path(claude_dir))
        else:
            paths = get_claude_paths()

        # Parse time filter
        time_filter = parse_time_filter(period)

        # Initialize session parser with session files
        session_parser = SessionParser(paths.get_project_session_files())

        # Count tool output tokens separately (Claude Code 2.1.0+)
        tool_output_tokens = session_parser.count_tool_output_tokens(time_filter=time_filter)

        # Get project stats
        project_stats = session_parser.get_project_stats(time_filter=time_filter)

        if not project_stats and tool_output_tokens == 0:
            console.print(f"\n[yellow]No usage data found for the last {period}[/yellow]")
            return

        # Build results list
        results = []
        for project_path, stats in sorted(project_stats.items(), key=lambda x: x[1].total_tokens.total_input_tokens, reverse=True):
            llm_tokens = stats.total_tokens.total_input_tokens
            total_tokens = llm_tokens + tool_output_tokens
            cost_usd = calculate_cost_from_tokens(total_tokens, tool_output_tokens)

            result = {
                'project': project_path,
                'sessions': stats.session_count,
                'messages': stats.message_count,
                'llm_tokens': llm_tokens,
                'tool_output_tokens': tool_output_tokens,
                'total_tokens': total_tokens,
                'cost_usd': round(cost_usd, 4),
            }
            results.append(result)

        # Output based on format
        if format == 'json':
            console.print_json(data=results)
        elif format == 'table':
            table = Table(title=f"Claude Usage by Project ({period}) - Includes Tool Outputs (2.1.0+)")
            table.add_column("Project", style="cyan")
            table.add_column("Sessions", justify="right")
            table.add_column("Messages", justify="right")
            table.add_column("LLM Tokens", justify="right")
            table.add_column("Tool Outputs", justify="right", style="dim")
            table.add_column("Total", justify="right", style="green")
            table.add_column("Est. Cost", justify="right")
            
            for r in results:
                table.add_row(
                    r['project'][:35] + '...' if len(r['project']) > 35 else r['project'],
                    str(r['sessions']),
                    str(r['messages']),
                    f"{r['llm_tokens']:,}",
                    f"{r['tool_output_tokens']:,}" if r['tool_output_tokens'] > 0 else "-",
                    f"{r['total_tokens']:,}",
                    f"${r['cost_usd']:.4f}"
                )
            console.print(table)
            
            # Add summary row
            total_llm = sum(r['llm_tokens'] for r in results)
            total_tool = sum(r['tool_output_tokens'] for r in results)
            total_cost = sum(r['cost_usd'] for r in results)
            
            summary_table = Table(show_header=False, box=None)
            summary_table.add_column("Metric", style="dim")
            summary_table.add_column("Value", justify="right")
            summary_table.add_row("LLM Tokens", f"{total_llm:,}")
            summary_table.add_row("Tool Output Tokens", f"{total_tool:,}")
            summary_table.add_row("Total Tokens", f"{total_llm + total_tool:,}")
            summary_table.add_row("Est. Cost", f"${total_cost:.4f}")
            console.print(summary_table)
        else:  # text format
            console.print(f"\n[cyan]Claude Usage by Project ({period}) - Includes Tool Outputs (2.1.0+)[/cyan]\n")

            for r in results:
                console.print(f"  [cyan]{r['project'][:50]}[/cyan]")
                console.print(f"    Sessions: {r['sessions']} | Messages: {r['messages']}")
                console.print(f"    LLM Tokens: {r['llm_tokens']:,} | Tool Outputs: {r['tool_output_tokens']:,}")
                console.print(f"    Total: {r['total_tokens']:,} tokens | Est. Cost: ${r['cost_usd']:.4f}\n")

    except FileNotFoundError as e:
        console.print(f"\n[red]✗ Error:[/red] {e}")
        console.print("\n[yellow]Make sure Claude Code has been used at least once.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]✗ Unexpected error:[/red] {e}")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


@cli.command(name='issue-agent')
@click.option(
    '--repo-owner',
    type=str,
    required=True,
    help='GitHub repository owner'
)
@click.option(
    '--repo-name',
    type=str,
    required=True,
    help='GitHub repository name'
)
@click.option(
    '--github-token',
    type=str,
    help='GitHub personal access token (uses GITHUB_TOKEN env var if not provided)'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Show which issue would be selected without taking action'
)
def issue_agent(repo_owner, repo_name, github_token, dry_run):
    """
    Run the GitHub issue agent to process the next issue.

    The agent will:
    1. Fetch all open issues from the repository
    2. Prioritize issues based on labels and age
    3. Select the highest priority unassigned issue
    4. (In future) Create a branch and PR to fix the issue

    Examples:

      # Run agent in dry-run mode to see which issue would be selected
      $ claudesavvy issue-agent --repo-owner allannapier --repo-name claudesavvy --dry-run

      # Run agent to actually process an issue
      $ claudesavvy issue-agent --repo-owner allannapier --repo-name claudesavvy

      # Use custom GitHub token
      $ claudesavvy issue-agent --repo-owner allannapier --repo-name claudesavvy --github-token ghp_xxx
    """
    console = Console()

    try:
        console.print("\n[cyan]═══ GitHub Issue Agent ═══[/cyan]")
        console.print(f"[dim]Repository: {repo_owner}/{repo_name}[/dim]\n")

        # Initialize agent
        agent = GitHubIssueAgent(repo_owner, repo_name, github_token)

        # Run agent
        if dry_run:
            console.print("[yellow]Running in dry-run mode...[/yellow]\n")

        result = agent.run()

        # Display results
        status = result['status']

        if status == 'no_issues':
            console.print("[yellow]✓ No open issues found in the repository[/yellow]\n")
        elif status == 'no_available_issues':
            console.print("[yellow]✓ No available issues to work on[/yellow]")
            console.print("[dim]All issues are either assigned or filtered out[/dim]\n")
        elif status == 'issue_selected':
            issue = result['issue']
            console.print("[green]✓ Selected issue to work on:[/green]\n")

            table = Table(show_header=False, box=None)
            table.add_column("Field", style="cyan")
            table.add_column("Value")

            table.add_row("Issue Number", f"#{issue['number']}")
            table.add_row("Title", issue['title'])
            table.add_row("Labels", ', '.join(issue['labels']) if issue['labels'] else 'None')
            table.add_row("URL", issue['url'])

            console.print(table)
            console.print()

            if dry_run:
                console.print("[dim]Dry-run mode: No branch or PR will be created[/dim]\n")
            else:
                console.print("[yellow]Note: Automatic branch and PR creation is not yet implemented[/yellow]")
                console.print("[dim]You can manually create a branch and PR for this issue[/dim]\n")
        else:
            console.print(f"[red]✗ Unknown status: {status}[/red]\n")

    except RuntimeError as e:
        console.print(f"\n[red]✗ Error:[/red] {e}\n")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]✗ Unexpected error:[/red] {e}")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


@cli.command(name='code-review')
@click.option(
    '--repo-owner',
    type=str,
    required=True,
    help='GitHub repository owner'
)
@click.option(
    '--repo-name',
    type=str,
    required=True,
    help='GitHub repository name'
)
@click.option(
    '--pr-number',
    type=int,
    help='Pull request number to review (if not provided, lists all open PRs)'
)
@click.option(
    '--github-token',
    type=str,
    help='GitHub personal access token (uses GITHUB_TOKEN env var if not provided)'
)
def code_review(repo_owner, repo_name, pr_number, github_token):
    """
    Run the GitHub code review agent to analyze pull requests.

    The agent will:
    1. Fetch pull request(s) from the repository
    2. Analyze code changes for security issues
    3. Check code quality and best practices
    4. Provide automated review comments

    Examples:

      # List all open pull requests
      $ claudesavvy code-review --repo-owner allannapier --repo-name claudesavvy

      # Review a specific pull request
      $ claudesavvy code-review --repo-owner allannapier --repo-name claudesavvy --pr-number 68

      # Use custom GitHub token
      $ claudesavvy code-review --repo-owner allannapier --repo-name claudesavvy --pr-number 68 --github-token ghp_xxx
    """
    console = Console()

    try:
        console.print("\n[cyan]═══ GitHub Code Review Agent ═══[/cyan]")
        console.print(f"[dim]Repository: {repo_owner}/{repo_name}[/dim]\n")

        # Initialize agent
        agent = GitHubCodeReviewAgent(repo_owner, repo_name, github_token)

        # Run agent
        result = agent.run(pr_number)

        # Display results
        status = result['status']

        if status == 'no_prs':
            console.print("[yellow]✓ No open pull requests found in the repository[/yellow]\n")
        elif status == 'pr_not_found':
            console.print(f"[red]✗ Pull request #{pr_number} not found[/red]\n")
        elif status == 'prs_listed':
            console.print("[green]✓ Open pull requests:[/green]\n")

            table = Table(show_header=True, box=None)
            table.add_column("PR #", style="cyan")
            table.add_column("Title")
            table.add_column("Author", style="dim")
            table.add_column("Files", justify="right", style="dim")
            table.add_column("+/-", justify="right", style="dim")

            for pr in result['prs']:
                table.add_row(
                    f"#{pr['number']}",
                    pr['title'][:50] + "..." if len(pr['title']) > 50 else pr['title'],
                    pr['author'],
                    str(pr['changed_files']),
                    f"+{pr['additions']}/-{pr['deletions']}"
                )

            console.print(table)
            console.print()
            console.print("[dim]Use --pr-number to review a specific pull request[/dim]\n")

        elif status == 'review_complete':
            pr = result['pr']
            review = result['review']

            console.print(f"[green]✓ Review complete for PR #{pr['number']}:[/green]\n")

            # PR Summary
            summary_table = Table(show_header=False, box=None)
            summary_table.add_column("Field", style="cyan")
            summary_table.add_column("Value")

            summary_table.add_row("Title", pr['title'])
            summary_table.add_row("Author", pr['author'])
            summary_table.add_row("Changed Files", str(pr['changed_files']))
            summary_table.add_row("Changes", f"+{pr['additions']} / -{pr['deletions']}")
            summary_table.add_row("URL", pr['url'])

            console.print(summary_table)
            console.print()

            # Review Summary
            console.print("[cyan]Review Summary:[/cyan]")
            console.print(f"  Total Comments: {review['total_comments']}")

            if review['errors'] > 0:
                console.print(f"  [red]Errors: {review['errors']}[/red]")
            if review['warnings'] > 0:
                console.print(f"  [yellow]Warnings: {review['warnings']}[/yellow]")
            if review['info'] > 0:
                console.print(f"  [blue]Info: {review['info']}[/blue]")

            if review['total_comments'] == 0:
                console.print("\n[green]✓ No issues found![/green]\n")
            else:
                console.print()

                # Group comments by severity
                errors = [c for c in review['comments'] if c['severity'] == 'error']
                warnings = [c for c in review['comments'] if c['severity'] == 'warning']
                info = [c for c in review['comments'] if c['severity'] == 'info']

                # Display errors
                if errors:
                    console.print("[red]Errors:[/red]")
                    for comment in errors:
                        console.print(f"  [red]✗[/red] {comment['file']}:{comment['line']}")
                        console.print(f"    {comment['comment']}\n")

                # Display warnings
                if warnings:
                    console.print("[yellow]Warnings:[/yellow]")
                    for comment in warnings:
                        console.print(f"  [yellow]⚠[/yellow] {comment['file']}:{comment['line']}")
                        console.print(f"    {comment['comment']}\n")

                # Display info
                if info:
                    console.print("[blue]Info:[/blue]")
                    for comment in info:
                        console.print(f"  [blue]ℹ[/blue] {comment['file']}:{comment['line']}")
                        console.print(f"    {comment['comment']}\n")

        else:
            console.print(f"[red]✗ Unknown status: {status}[/red]\n")

    except RuntimeError as e:
        console.print(f"\n[red]✗ Error:[/red] {e}\n")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]✗ Unexpected error:[/red] {e}")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


# For backward compatibility, also expose main
main = serve


if __name__ == '__main__':
    cli()
