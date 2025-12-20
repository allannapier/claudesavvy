"""Main CLI interface for claude-monitor."""

import sys
import click
from pathlib import Path
from rich.console import Console

from .utils.paths import get_claude_paths, ClaudeDataPaths
from .utils.time_filter import TimeFilter
from .parsers.history import HistoryParser
from .parsers.sessions import SessionParser
from .parsers.debug import DebugLogParser
from .parsers.files import FileHistoryParser
from .parsers.tools import ToolUsageParser
from .parsers.skills import SkillsParser, ConfigurationParser
from .analyzers.usage import UsageAnalyzer
from .analyzers.tokens import TokenAnalyzer
from .analyzers.integrations import IntegrationAnalyzer
from .analyzers.features import FeaturesAnalyzer
from .display.dashboard import Dashboard
from .display.menu import InteractiveMenu


def create_dashboard(
    time_filter: TimeFilter,
    paths: ClaudeDataPaths
) -> Dashboard:
    """
    Create a dashboard instance with all analyzers.

    Args:
        time_filter: TimeFilter instance
        paths: ClaudeDataPaths instance

    Returns:
        Dashboard instance
    """
    # Initialize parsers
    history_parser = HistoryParser(paths.history_file)
    session_parser = SessionParser(paths.get_project_session_files())

    # Build project map for debug logs (session_id -> project_name)
    # First, build a list of all encoded project paths to find the longest matches
    project_paths = {}
    if paths.projects_dir.exists():
        all_dirs = sorted([d for d in paths.projects_dir.iterdir() if d.is_dir()],
                         key=lambda x: len(x.name), reverse=True)

        for project_dir in all_dirs:
            encoded_name = project_dir.name
            # For each session file, find the most specific project directory
            for session_file in project_dir.glob("*.jsonl"):
                session_id = session_file.stem
                if session_id not in project_paths or len(encoded_name) > len(project_paths[session_id]):
                    # Remove common prefix and extract project name
                    # Common pattern: -Users-allannapier-code-PROJECT_NAME
                    if encoded_name.startswith('-Users-allannapier-code-'):
                        project_name = encoded_name.replace('-Users-allannapier-code-', '', 1)
                    elif encoded_name.startswith('-'):
                        # Fallback: just remove leading dash
                        project_name = encoded_name[1:]
                    else:
                        project_name = encoded_name
                    project_paths[session_id] = project_name

    project_map = project_paths

    debug_parser = DebugLogParser(paths.get_debug_log_files(), project_map)
    file_parser = FileHistoryParser(paths.file_history_dir, paths.get_project_session_files())
    tool_parser = ToolUsageParser(paths.get_project_session_files())
    skills_parser = SkillsParser(paths.base_dir / "skills")
    config_parser = ConfigurationParser(paths.base_dir)

    # Initialize analyzers
    usage_analyzer = UsageAnalyzer(history_parser, session_parser, time_filter)
    token_analyzer = TokenAnalyzer(session_parser, time_filter)
    integration_analyzer = IntegrationAnalyzer(debug_parser)
    features_analyzer = FeaturesAnalyzer(
        tool_parser, skills_parser, debug_parser, config_parser, time_filter
    )

    # Create dashboard
    return Dashboard(
        usage_analyzer=usage_analyzer,
        token_analyzer=token_analyzer,
        integration_analyzer=integration_analyzer,
        features_analyzer=features_analyzer,
        file_parser=file_parser
    )


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """
    Claude Code Usage Tracking Tool

    Display comprehensive metrics about your Claude Code usage including
    sessions, token consumption, costs, projects, integrations, and more.

    Use 'claude-monitor COMMAND --help' for more information on a command.

    Examples:

      # Show dashboard (default)
      $ claude-monitor

      # Show web interface
      $ claude-monitor web

      # Show today's activity
      $ claude-monitor dashboard --today

      # Start web server on custom port
      $ claude-monitor web --port 8080
    """
    # If no command is provided, default to dashboard
    if ctx.invoked_subcommand is None:
        ctx.invoke(dashboard)


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
def web(port, host, debug, claude_dir):
    """
    Run Claude Monitor web server.

    Start a Flask development server to view Claude Code usage metrics
    in a web browser. Access the dashboard at http://localhost:5000

    Examples:

      # Start server on default port 5000
      $ claude-monitor web

      # Start on custom port with debug mode
      $ claude-monitor web --port 8080 --debug

      # Bind to all network interfaces
      $ claude-monitor web --host 0.0.0.0

      # Use custom Claude data directory
      $ claude-monitor web --claude-dir /path/to/claude/data
    """
    console = Console()

    try:
        # Lazy import Flask to avoid requiring it when not using web command
        from .web.app import create_app

        # Get Claude data paths
        if claude_dir:
            paths = ClaudeDataPaths(Path(claude_dir))
        else:
            paths = get_claude_paths()

        # Create Flask app with paths
        app = create_app(paths)

        # Display startup information
        console.print(f"\n[cyan]Claude Monitor Web Server[/cyan]")
        console.print(f"[dim]Starting web server...[/dim]\n")
        console.print(f"[green]Web server running at:[/green] [bold]http://{host}:{port}[/bold]")
        if debug:
            console.print(f"[yellow]Debug mode:[/yellow] [bold]Enabled[/bold]")
        console.print(f"\n[dim]Press Ctrl+C to stop the server[/dim]\n")

        # Run Flask app
        app.run(host=host, port=port, debug=debug)

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}", style="red")
        console.print("\n[yellow]Make sure Claude Code has been used at least once.[/yellow]")
        sys.exit(1)

    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}", style="red")
        console.print("\n[dim]If this issue persists, please report it.[/dim]")
        sys.exit(1)


@click.command()
@click.option(
    '--today',
    'time_preset',
    flag_value='today',
    help='Show only today\'s activity'
)
@click.option(
    '--week',
    'time_preset',
    flag_value='week',
    help='Show last 7 days of activity'
)
@click.option(
    '--month',
    'time_preset',
    flag_value='month',
    help='Show last 30 days of activity'
)
@click.option(
    '--quarter',
    'time_preset',
    flag_value='quarter',
    help='Show current quarter activity'
)
@click.option(
    '--year',
    'time_preset',
    flag_value='year',
    help='Show last year of activity'
)
@click.option(
    '--all',
    'time_preset',
    flag_value='all',
    help='Show all-time activity (default)'
)
@click.option(
    '--since',
    type=str,
    help='Show activity since a specific date (YYYY-MM-DD)'
)
@click.option(
    '--focus',
    type=click.Choice(['features'], case_sensitive=False),
    help='Focus on a specific metric category'
)
@click.option(
    '--project',
    type=str,
    help='Filter by specific project path'
)
@click.option(
    '--claude-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help='Custom Claude data directory (default: ~/.claude/)'
)
@click.option(
    '--interactive/--no-interactive',
    'interactive',
    default=None,
    help='Enable interactive menu mode (default when no options provided)'
)
def dashboard(time_preset, since, focus, project, claude_dir, interactive):
    """
    Claude Code Usage Tracking Tool

    Display comprehensive metrics about your Claude Code usage including
    sessions, token consumption, costs, projects, integrations, and more.

    Examples:

      # Show all-time dashboard (interactive menu)
      $ claude-monitor

      # Show today's activity
      $ claude-monitor --today

      # Show current quarter's activity
      $ claude-monitor --quarter

      # Show last week with focus on features
      $ claude-monitor --week --focus features

      # Show activity since a specific date
      $ claude-monitor --since 2025-01-01

      # Filter by project
      $ claude-monitor --project /path/to/project
    """
    console = Console()

    try:
        # Get Claude data paths
        if claude_dir:
            paths = ClaudeDataPaths(Path(claude_dir))
        else:
            paths = get_claude_paths()

        # Check if we should show interactive menu
        # Show interactive menu if:
        # 1. --interactive flag is explicitly set, OR
        # 2. No CLI options were provided (default behavior)
        has_cli_options = any([since, focus, project, time_preset is not None])
        show_interactive = interactive is True or (interactive is None and not has_cli_options)

        if show_interactive:
            menu = InteractiveMenu(console)

            # Interactive loop - keep showing menu until user quits
            while True:
                view_choice, timeframe_choice = menu.get_user_choice()

                if view_choice == "quit":
                    console.print("\n[cyan]Thanks for using Claude Monitor! 👋[/cyan]")
                    sys.exit(0)

                # Map choices to options
                options = menu.map_choices(view_choice, timeframe_choice)
                focus = options['focus']
                time_preset = options['time_preset']

                # Create time filter
                preset = time_preset if time_preset else 'all'
                time_filter = TimeFilter.from_preset(preset)

                # Create dashboard
                dashboard = create_dashboard(time_filter, paths)

                # Display based on focus option
                if focus == 'features':
                    dashboard.display_features_only()
                else:
                    dashboard.display_full_dashboard()

                # Prompt to return to menu or quit
                console.print("\n")
                console.print("[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]")
                console.print("[white][[cyan]M[/cyan]] Return to Menu  |  [[red]Q[/red]] Quit[/white]")
                from rich.prompt import Prompt
                action = Prompt.ask(
                    "",
                    choices=["m", "M", "q", "Q"],
                    default="m"
                )

                if action.upper() == "Q":
                    console.print("\n[cyan]Thanks for using Claude Monitor! 👋[/cyan]")
                    sys.exit(0)

                # Return to menu (M) - loop continues
                console.clear()
        else:
            # Non-interactive mode - single execution
            # Create time filter
            if since:
                time_filter = TimeFilter.from_since(since)
            else:
                # Use time_preset if provided, otherwise default to 'all'
                preset = time_preset if time_preset else 'all'
                time_filter = TimeFilter.from_preset(preset)

            # Create dashboard
            dashboard = create_dashboard(time_filter, paths)

            # Display based on focus option
            if focus == 'features':
                dashboard.display_features_only()
            else:
                dashboard.display_full_dashboard()

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}", style="red")
        console.print("\n[yellow]Make sure Claude Code has been used at least once.[/yellow]")
        sys.exit(1)

    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}", style="red")
        console.print("\n[dim]If this issue persists, please report it.[/dim]")
        sys.exit(1)


# Register commands with the group
main.add_command(dashboard)
main.add_command(web)


if __name__ == '__main__':
    main()
