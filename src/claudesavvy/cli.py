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
@click.option(
    '--demo',
    is_flag=True,
    help='Run with sample data (no Claude Code installation required)'
)
def serve(port, host, debug, claude_dir, demo):
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
        if demo:
            from .data.demo import generate_demo_data
            paths = generate_demo_data()
        elif claude_dir:
            paths = ClaudeDataPaths(Path(claude_dir))
        else:
            paths = get_claude_paths()

        # Create Flask app with paths
        app = create_app(paths)

        # Display startup information
        console.print("\n[cyan]═══ ClaudeSavvy Web Server ═══[/cyan]")
        console.print("[dim]Starting web interface...[/dim]\n")
        console.print(f"[green]✓ Server running at:[/green] [bold]http://{host}:{port}[/bold]")

        if demo:
            console.print("[yellow]⚠ Demo mode:[/yellow] [bold]Showing sample data[/bold] (not your real usage)")

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


# For backward compatibility, also expose main
main = serve


if __name__ == '__main__':
    cli()
