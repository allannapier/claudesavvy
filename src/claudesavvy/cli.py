"""Main CLI interface for claudesavvy."""

import sys
import click
from pathlib import Path
from rich.console import Console

from .utils.paths import get_claude_paths, ClaudeDataPaths


@click.command()
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
def main(port, host, debug, claude_dir):
    """
    ClaudeSavvy - Web-based usage tracking for Claude Code

    Visualize your Claude Code usage metrics including sessions, token consumption,
    costs, projects, file operations, and MCP integrations through an intuitive
    web dashboard.

    Access the dashboard at http://localhost:5000 after starting the server.

    Examples:

      # Start server on default port 5000
      $ claudesavvy

      # Start on custom port
      $ claudesavvy --port 8080

      # Bind to all network interfaces (allow external access)
      $ claudesavvy --host 0.0.0.0

      # Enable debug mode with auto-reload
      $ claudesavvy --debug

      # Use custom Claude data directory
      $ claudesavvy --claude-dir /path/to/claude/data
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
        console.print("https://github.com/allannapier/claude_monitor/issues[/dim]\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
