"""Dashboard route handler for Claude Monitor web application."""

from typing import Any, Dict
from flask import Blueprint, render_template, current_app
import logging

logger = logging.getLogger(__name__)

dashboard_bp = Blueprint('dashboard', __name__)


@dashboard_bp.route('/')
@dashboard_bp.route('/dashboard')
def index() -> str:
    """
    Render the main dashboard page with usage statistics and project breakdown.

    Fetches data from the DashboardService and passes it to the dashboard template.

    Returns:
        Rendered HTML template with dashboard data or error page

    Raises:
        RuntimeError: If dashboard_service is not available on the app
    """
    try:
        # Get the dashboard service from the app context
        if not hasattr(current_app, 'dashboard_service'):
            logger.error('Dashboard service not found on current app')
            return render_template(
                'pages/error.html',
                error_title='Configuration Error',
                error_message='Dashboard service not configured. Please restart the server.',
                error_details='The dashboard service was not properly initialized.'
            ), 500

        service = current_app.dashboard_service

        # Fetch all dashboard data with error handling
        usage_summary: Dict[str, Any] = service.get_usage_summary()
        token_summary: Dict[str, Any] = service.get_token_summary()
        project_breakdown: Dict[str, Any] = service.get_project_breakdown()

        logger.debug('Dashboard data fetched successfully')

        # Render template with all data
        return render_template(
            'pages/dashboard.html',
            usage=usage_summary,
            tokens=token_summary,
            projects=project_breakdown
        )

    except ValueError as e:
        # Handle data parsing errors (like invalid timestamps)
        logger.error(f'Data parsing error: {e}', exc_info=True)
        return render_template(
            'pages/error.html',
            error_title='Data Parsing Error',
            error_message='Unable to parse Claude Code usage data.',
            error_details=f'There appears to be corrupted or invalid data in your ~/.claude/ directory. Error: {str(e)}',
            suggestion='Try using the CLI tool to see if it shows more details: claude-monitor'
        ), 500

    except FileNotFoundError as e:
        # Handle missing data directory
        logger.error(f'Claude data not found: {e}')
        return render_template(
            'pages/error.html',
            error_title='No Claude Data Found',
            error_message='Unable to find Claude Code usage data.',
            error_details='Make sure you have used Claude Code at least once before running the monitor.',
            suggestion='Run Claude Code first, then come back here to view your usage statistics.'
        ), 404

    except Exception as e:
        # Handle any other errors
        logger.error(f'Unexpected error loading dashboard: {e}', exc_info=True)
        return render_template(
            'pages/error.html',
            error_title='Unexpected Error',
            error_message='An unexpected error occurred while loading the dashboard.',
            error_details=str(e),
            suggestion='Please check the server logs for more details or report this issue.'
        ), 500


@dashboard_bp.route('/tokens')
def tokens() -> str:
    """
    Render the tokens page with detailed token usage breakdown.

    Returns:
        Rendered HTML template with token data or error page
    """
    try:
        service = current_app.dashboard_service
        token_summary: Dict[str, Any] = service.get_token_summary()

        return render_template(
            'pages/tokens.html',
            tokens=token_summary
        )

    except ValueError as e:
        logger.error(f'Data parsing error: {e}', exc_info=True)
        return render_template(
            'pages/error.html',
            error_title='Data Parsing Error',
            error_message='Unable to parse Claude Code usage data.',
            error_details=f'Error: {str(e)}',
            suggestion='Try using the CLI tool: claude-monitor'
        ), 500

    except Exception as e:
        logger.error(f'Unexpected error loading tokens page: {e}', exc_info=True)
        return render_template(
            'pages/error.html',
            error_title='Unexpected Error',
            error_message='An unexpected error occurred.',
            error_details=str(e)
        ), 500


@dashboard_bp.route('/projects')
def projects() -> str:
    """
    Render the projects page with per-project analytics.

    Returns:
        Rendered HTML template with project data or error page
    """
    try:
        service = current_app.dashboard_service
        project_breakdown: Dict[str, Any] = service.get_project_breakdown()

        return render_template(
            'pages/projects.html',
            projects=project_breakdown
        )

    except ValueError as e:
        logger.error(f'Data parsing error: {e}', exc_info=True)
        return render_template(
            'pages/error.html',
            error_title='Data Parsing Error',
            error_message='Unable to parse Claude Code usage data.',
            error_details=f'Error: {str(e)}',
            suggestion='Try using the CLI tool: claude-monitor'
        ), 500

    except Exception as e:
        logger.error(f'Unexpected error loading projects page: {e}', exc_info=True)
        return render_template(
            'pages/error.html',
            error_title='Unexpected Error',
            error_message='An unexpected error occurred.',
            error_details=str(e)
        ), 500


@dashboard_bp.route('/files')
def files() -> str:
    """Render the files page with file operation statistics."""
    try:
        service = current_app.dashboard_service

        # Get file statistics
        file_data = service.get_file_statistics(limit=20)

        return render_template('pages/files.html', files=file_data)

    except Exception as e:
        logger.error(f'Error loading files page: {e}', exc_info=True)
        return render_template('pages/files.html', files={})


@dashboard_bp.route('/integrations')
def integrations() -> str:
    """Render the integrations page with MCP server statistics."""
    try:
        service = current_app.dashboard_service

        # Get MCP integration data
        mcp_data = service.get_mcp_integrations()

        # Format data for template
        integrations_data = {
            'active_count': mcp_data['total_servers'],
            'most_used_ide': 'CLI',  # Claude Code is CLI-based
            'mcp_servers': mcp_data['total_servers'],
            'total_calls': mcp_data['total_calls'],
            'most_used_server': mcp_data['most_used_server'],
            'servers': mcp_data['servers']
        }

        return render_template('pages/integrations.html', integrations=integrations_data)

    except Exception as e:
        logger.error(f'Error loading integrations page: {e}', exc_info=True)
        return render_template('pages/integrations.html', integrations={
            'most_used_ide': 'CLI',
            'servers': []
        })


@dashboard_bp.route('/features')
def features() -> str:
    """Render the features page with tool usage statistics."""
    try:
        service = current_app.dashboard_service

        # Get tool usage data
        top_tools = service.get_top_tools(limit=10)

        # Calculate summary stats
        total_tools = len(top_tools.get('tools', []))
        total_calls = sum(tool.get('invocation_count', 0) for tool in top_tools.get('tools', []))
        most_used_tool = top_tools['tools'][0]['tool_name'] if top_tools.get('tools') else 'N/A'

        features_data = {
            'total_tools': total_tools,
            'total_tool_calls': total_calls,
            'most_used_tool': most_used_tool,
            'top_tools': top_tools.get('tools', [])
        }

        return render_template('pages/features.html', features=features_data)

    except Exception as e:
        logger.error(f'Error loading features page: {e}', exc_info=True)
        return render_template('pages/features.html', features={})


@dashboard_bp.route('/api/dashboard')
def api_dashboard() -> str:
    """
    API endpoint for HTMX to fetch filtered dashboard data.
    
    Query params:
        period: today|week|month|all (default: all)
    
    Returns:
        Rendered HTML partial with dashboard content
    """
    from flask import request
    from ...utils.time_filter import TimeFilter
    from datetime import datetime, timedelta
    
    try:
        service = current_app.dashboard_service
        period = request.args.get('period', 'all')
        
        # Create time filter based on period
        time_filter = None
        if period != 'all':
            now = datetime.now()
            if period == 'today':
                start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            elif period == 'week':
                start_time = now - timedelta(days=7)
            elif period == 'month':
                start_time = now - timedelta(days=30)
            else:
                start_time = None
            
            if start_time:
                time_filter = TimeFilter(start_time=start_time, end_time=now)
        
        # Fetch filtered data
        usage_summary: Dict[str, Any] = service.get_usage_summary(time_filter=time_filter)
        token_summary: Dict[str, Any] = service.get_token_summary(time_filter=time_filter)
        project_breakdown: Dict[str, Any] = service.get_project_breakdown(time_filter=time_filter)
        
        # Render partial template
        return render_template(
            'partials/dashboard_content.html',
            usage=usage_summary,
            tokens=token_summary,
            projects=project_breakdown
        )
    
    except Exception as e:
        logger.error(f'Error loading filtered dashboard: {e}', exc_info=True)
        return f'<div class="text-red-600 p-4">Error loading data: {str(e)}</div>', 500


@dashboard_bp.route('/export/<format>')
def export_data(format: str) -> Any:
    """
    Export dashboard data in various formats.
    
    Args:
        format: csv or json
    
    Returns:
        Downloaded file
    """
    import csv
    import io
    from flask import make_response
    
    try:
        service = current_app.dashboard_service
        
        if format not in ['csv', 'json']:
            return "Invalid format. Use 'csv' or 'json'", 400
        
        # Fetch all data
        usage_summary = service.get_usage_summary()
        token_summary = service.get_token_summary()
        project_breakdown = service.get_project_breakdown()
        
        if format == 'json':
            # Export as JSON
            from flask import jsonify
            data = {
                'usage': usage_summary,
                'tokens': token_summary,
                'projects': project_breakdown,
                'exported_at': datetime.now().isoformat()
            }
            response = make_response(jsonify(data))
            response.headers['Content-Disposition'] = 'attachment; filename=claude_monitor_export.json'
            response.headers['Content-Type'] = 'application/json'
            return response
        
        elif format == 'csv':
            # Export as CSV
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write usage summary
            writer.writerow(['Usage Summary'])
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Commands', usage_summary.get('total_commands', 0)])
            writer.writerow(['Total Sessions', usage_summary.get('total_sessions', 0)])
            writer.writerow(['Total Messages', usage_summary.get('total_messages', 0)])
            writer.writerow(['Active Projects', usage_summary.get('total_projects', 0)])
            writer.writerow([])
            
            # Write token summary
            writer.writerow(['Token Summary'])
            writer.writerow(['Token Type', 'Count', 'Cost'])
            writer.writerow(['Input Tokens', token_summary.get('input_tokens', 0), f"${token_summary.get('input_cost', 0):.2f}"])
            writer.writerow(['Output Tokens', token_summary.get('output_tokens', 0), f"${token_summary.get('output_cost', 0):.2f}"])
            writer.writerow(['Cache Creation', token_summary.get('cache_creation_tokens', 0), f"${token_summary.get('cache_creation_cost', 0):.2f}"])
            writer.writerow(['Cache Reads', token_summary.get('cache_read_tokens', 0), f"${token_summary.get('cache_read_cost', 0):.2f}"])
            writer.writerow(['Total Cost', '-', f"${token_summary.get('total_cost', 0):.2f}"])
            writer.writerow([])
            
            # Write projects
            if project_breakdown.get('projects'):
                writer.writerow(['Projects'])
                writer.writerow(['Name', 'Commands', 'Sessions', 'Messages', 'Total Tokens', 'Cost'])
                for project in project_breakdown['projects']:
                    writer.writerow([
                        project.get('name', ''),
                        project.get('command_count', 0),
                        project.get('session_count', 0),
                        project.get('message_count', 0),
                        project.get('total_tokens', 0),
                        f"${project.get('total_cost', 0):.2f}"
                    ])
            
            response = make_response(output.getvalue())
            response.headers['Content-Disposition'] = 'attachment; filename=claude_monitor_export.csv'
            response.headers['Content-Type'] = 'text/csv'
            return response
    
    except Exception as e:
        logger.error(f'Error exporting data: {e}', exc_info=True)
        return f'Error exporting data: {str(e)}', 500
