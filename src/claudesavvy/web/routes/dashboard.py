"""Dashboard route handler for Claude Monitor web application."""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from flask import Blueprint, render_template, current_app
import logging

from ...utils.time_filter import TimeFilter

logger = logging.getLogger(__name__)

dashboard_bp = Blueprint("dashboard", __name__)


def get_time_filter_from_period(period: str) -> Optional[TimeFilter]:
    """
    Create a TimeFilter from a period string.

    Args:
        period: One of '15min', '1hour', 'today', 'week', 'month', or 'all'

    Returns:
        TimeFilter instance or None for 'all'
    """
    if period == "all":
        return None

    now = datetime.now()
    start_time = None

    if period == "15min":
        start_time = now - timedelta(minutes=15)
    elif period == "1hour":
        start_time = now - timedelta(hours=1)
    elif period == "today":
        start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "week":
        start_time = now - timedelta(days=7)
    elif period == "month":
        start_time = now - timedelta(days=30)

    if start_time:
        return TimeFilter(start_time=start_time, end_time=now)
    return None


@dashboard_bp.route("/")
@dashboard_bp.route("/dashboard")
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
        if not hasattr(current_app, "dashboard_service"):
            logger.error("Dashboard service not found on current app")
            return render_template(
                "pages/error.html",
                error_title="Configuration Error",
                error_message="Dashboard service not configured. Please restart the server.",
                error_details="The dashboard service was not properly initialized.",
            ), 500

        service = current_app.dashboard_service

        # Default to 1 hour filter for faster initial load
        time_filter = get_time_filter_from_period("1hour")

        # Fetch all dashboard data with error handling
        usage_summary: Dict[str, Any] = service.get_usage_summary(
            time_filter=time_filter
        )
        token_summary: Dict[str, Any] = service.get_token_summary(
            time_filter=time_filter
        )
        project_breakdown: Dict[str, Any] = service.get_project_breakdown(
            time_filter=time_filter
        )
        model_breakdown: Dict[str, Any] = service.get_model_breakdown(
            time_filter=time_filter
        )
        token_trend: Dict[str, Any] = service.get_daily_token_trend(
            days=7, time_filter=time_filter
        )
        cost_trend: Dict[str, Any] = service.get_daily_cost_trend(
            days=7, time_filter=time_filter
        )
        project_cost_trend: Dict[str, Any] = service.get_project_cost_trend(
            days=7, max_projects=8, time_filter=time_filter
        )

        logger.debug("Dashboard data fetched successfully")

        # Render template with all data
        return render_template(
            "pages/dashboard.html",
            usage=usage_summary,
            tokens=token_summary,
            projects=project_breakdown,
            models=model_breakdown,
            token_trend=token_trend,
            cost_trend=cost_trend,
            project_cost_trend=project_cost_trend,
        )

    except ValueError as e:
        # Handle data parsing errors (like invalid timestamps)
        logger.error(f"Data parsing error: {e}", exc_info=True)
        return render_template(
            "pages/error.html",
            error_title="Data Parsing Error",
            error_message="Unable to parse Claude Code usage data.",
            error_details="There appears to be corrupted or invalid data in your ~/.claude/ directory.",
            suggestion="Try using the CLI tool to see if it shows more details: claude-monitor",
        ), 500

    except FileNotFoundError as e:
        # Handle missing data directory
        logger.error(f"Claude data not found: {e}")
        return render_template(
            "pages/error.html",
            error_title="No Claude Data Found",
            error_message="Unable to find Claude Code usage data.",
            error_details="Make sure you have used Claude Code at least once before running the monitor.",
            suggestion="Run Claude Code first, then come back here to view your usage statistics.",
        ), 404

    except Exception as e:
        # Handle any other errors
        logger.error(f"Unexpected error loading dashboard: {e}", exc_info=True)
        return render_template(
            "pages/error.html",
            error_title="Unexpected Error",
            error_message="An unexpected error occurred while loading the dashboard.",
            error_details="Check the server logs for more details.",
            suggestion="Please check the server logs for more details or report this issue.",
        ), 500


@dashboard_bp.route("/tokens")
def tokens() -> str:
    """
    Render the tokens page with detailed token usage breakdown.

    Returns:
        Rendered HTML template with token data or error page
    """
    try:
        service = current_app.dashboard_service
        # Default to 1hour filter on initial page load
        default_period = "1hour"
        time_filter = get_time_filter_from_period(default_period)
        token_summary: Dict[str, Any] = service.get_token_summary(
            time_filter=time_filter
        )
        model_breakdown: Dict[str, Any] = service.get_model_breakdown(
            time_filter=time_filter
        )
        available_models = service.get_available_models(time_filter=time_filter)

        return render_template(
            "pages/tokens.html",
            tokens=token_summary,
            models=model_breakdown,
            available_models=available_models,
            period=default_period,
        )

    except ValueError as e:
        logger.error(f"Data parsing error: {e}", exc_info=True)
        return render_template(
            "pages/error.html",
            error_title="Data Parsing Error",
            error_message="Unable to parse Claude Code usage data.",
            error_details="Check the server logs for details.",
            suggestion="Try using the CLI tool: claude-monitor",
        ), 500

    except Exception as e:
        logger.error(f"Unexpected error loading tokens page: {e}", exc_info=True)
        return render_template(
            "pages/error.html",
            error_title="Unexpected Error",
            error_message="An unexpected error occurred.",
            error_details="Check the server logs for details.",
        ), 500


@dashboard_bp.route("/projects")
def projects() -> str:
    """
    Render the projects page with per-project analytics.

    Returns:
        Rendered HTML template with project data or error page
    """
    try:
        service = current_app.dashboard_service
        # Default to 1hour filter on initial page load
        default_period = "1hour"
        time_filter = get_time_filter_from_period(default_period)
        project_breakdown: Dict[str, Any] = service.get_project_breakdown(
            time_filter=time_filter
        )
        available_models = service.get_available_models(time_filter=time_filter)

        return render_template(
            "pages/projects.html",
            projects=project_breakdown,
            available_models=available_models,
            period=default_period,
        )

    except ValueError as e:
        logger.error(f"Data parsing error: {e}", exc_info=True)
        return render_template(
            "pages/error.html",
            error_title="Data Parsing Error",
            error_message="Unable to parse Claude Code usage data.",
            error_details="Check the server logs for details.",
            suggestion="Try using the CLI tool: claude-monitor",
        ), 500

    except Exception as e:
        logger.error(f"Unexpected error loading projects page: {e}", exc_info=True)
        return render_template(
            "pages/error.html",
            error_title="Unexpected Error",
            error_message="An unexpected error occurred.",
            error_details="Check the server logs for details.",
        ), 500


@dashboard_bp.route("/files")
def files() -> str:
    """Render the files page with file operation statistics."""
    try:
        service = current_app.dashboard_service
        # Default to 1hour filter on initial page load
        default_period = "1hour"
        time_filter = get_time_filter_from_period(default_period)

        # Get file statistics
        file_data = service.get_file_statistics(limit=20, time_filter=time_filter)

        return render_template(
            "pages/files.html", files=file_data, period=default_period
        )

    except Exception as e:
        logger.error(f"Error loading files page: {e}", exc_info=True)
        return render_template("pages/files.html", files={})


@dashboard_bp.route("/integrations")
def integrations() -> str:
    """Render the integrations page with MCP server statistics."""
    try:
        service = current_app.dashboard_service
        # Default to 1hour filter on initial page load
        default_period = "1hour"
        time_filter = get_time_filter_from_period(default_period)

        # Get MCP integration data
        mcp_data = service.get_mcp_integrations(time_filter=time_filter)
        available_models = service.get_available_models(time_filter=time_filter)

        # Format data for template
        integrations_data = {
            "active_count": mcp_data["total_servers"],
            "most_used_ide": "CLI",  # Claude Code is CLI-based
            "mcp_servers": mcp_data["total_servers"],
            "total_calls": mcp_data["total_calls"],
            "total_tokens": mcp_data.get("total_tokens", 0),
            "total_cost": mcp_data.get("total_cost", 0.0),
            "most_used_server": mcp_data["most_used_server"],
            "server_activity": mcp_data["servers"],
        }

        return render_template(
            "pages/integrations.html",
            integrations=integrations_data,
            available_models=available_models,
            period=default_period,
        )

    except Exception as e:
        logger.error(f"Error loading integrations page: {e}", exc_info=True)
        return render_template(
            "pages/integrations.html",
            integrations={"most_used_ide": "CLI", "server_activity": []},
        )


@dashboard_bp.route("/features")
def features() -> str:
    """Render the features page with tool usage statistics."""
    try:
        service = current_app.dashboard_service
        # Default to 1hour filter on initial page load
        default_period = "1hour"
        time_filter = get_time_filter_from_period(default_period)

        # Get tool usage data
        top_tools = service.get_top_tools(limit=20, time_filter=time_filter)
        features_summary = service.get_features_summary(time_filter=time_filter)

        # Build tool_usage list with proper field names for template
        tool_usage = []
        for tool in top_tools.get("tools", []):
            tool_usage.append(
                {
                    "name": tool.get("tool_name", ""),
                    "category": tool.get("category", "tool"),
                    "usage_count": tool.get("invocation_count", 0),
                    "total_tokens": tool.get("total_tokens", 0),
                    "cost": tool.get("total_cost", 0.0),
                    "cost_per_call": tool.get("cost_per_call", 0.0),
                    "trend": None,  # No trends on initial load
                }
            )

        features_data = {
            "total_tools": len(tool_usage),
            "total_calls": top_tools.get("total_calls", 0),
            "total_tokens": top_tools.get("total_tokens", 0),
            "total_cost": top_tools.get("total_cost", 0.0),
            "sub_agents_used": features_summary.get("subagents", {}).get(
                "unique_used", 0
            ),
            "tool_usage": tool_usage,
            "categories": features_summary.get("categories", {}),
            "has_trends": False,  # No trends on initial load
        }

        return render_template(
            "pages/features.html", features=features_data, period=default_period
        )

    except Exception as e:
        logger.error(f"Error loading features page: {e}", exc_info=True)
        return render_template("pages/features.html", features={})


@dashboard_bp.route("/configuration")
def configuration() -> str:
    """Render the configuration page with Claude Code configuration viewer."""
    from flask import request

    try:
        service = current_app.dashboard_service

        # Get discovered repositories
        repositories = service.get_discovered_repositories()

        # Get selected repo from query params, default to first (User Configuration)
        selected_repo_path = request.args.get(
            "repo", repositories[0]["path"] if repositories else None
        )

        # Find the selected repository
        selected_repo = None
        if selected_repo_path:
            for repo in repositories:
                if repo["path"] == selected_repo_path:
                    selected_repo = repo
                    break

        # Default to first repo if not found
        if not selected_repo and repositories:
            selected_repo = repositories[0]
            selected_repo_path = repositories[0]["path"]

        # Get features for selected repository
        features = {}
        if selected_repo_path:
            features = service.get_configuration_features(selected_repo_path)

        return render_template(
            "pages/configuration.html",
            repositories=repositories,
            selected_repo=selected_repo,
            features=features,
        )

    except Exception as e:
        logger.error(f"Error loading configuration page: {e}", exc_info=True)
        return render_template(
            "pages/configuration.html", repositories=[], selected_repo=None, features={}
        )


@dashboard_bp.route("/api/configuration/<path:repo_path>")
def api_configuration(repo_path: str) -> str:
    """
    API endpoint for HTMX to fetch configuration features for a specific repository.

    Args:
        repo_path: Path to the repository

    Returns:
        Rendered HTML partial with feature list
    """
    try:
        service = current_app.dashboard_service
        # Flask's path converter strips leading slash, so add it back if missing
        if not repo_path.startswith("/"):
            repo_path = "/" + repo_path
        features = service.get_configuration_features(repo_path)

        return render_template("partials/feature_list.html", features=features)

    except Exception:
        # Sanitize user-provided repo_path to prevent log injection
        safe_repo_path = repo_path.replace("\r", "").replace("\n", "")
        logger.error(
            f"Error loading configuration for repo {safe_repo_path}",
            exc_info=True,
        )
        return render_template("partials/feature_list.html", features={})


@dashboard_bp.route("/api/feature-detail/<path:repo_path>/<feature_type>/<feature_id>")
def api_feature_detail(repo_path: str, feature_type: str, feature_id: str) -> str:
    """
    API endpoint for HTMX to fetch detailed information about a specific feature.

    Args:
        repo_path: Path to the repository
        feature_type: Type of feature (skill, mcp, command, plugin, hook, agent)
        feature_id: Name/ID of the feature

    Returns:
        Rendered HTML partial with feature details
    """
    try:
        service = current_app.dashboard_service
        # Flask's path converter strips leading slash, so add it back if missing
        if not repo_path.startswith("/"):
            repo_path = "/" + repo_path
        detail = service.get_feature_detail(repo_path, feature_type, feature_id)

        # Render appropriate detail template based on feature type
        template_map = {
            "skill": "partials/details/skill_detail.html",
            "mcp": "partials/details/mcp_detail.html",
            "command": "partials/details/command_detail.html",
            "plugin": "partials/details/plugin_detail.html",
            "hook": "partials/details/hook_detail.html",
            "agent": "partials/details/agent_detail.html",
        }

        template = template_map.get(
            feature_type, "partials/details/feature_detail.html"
        )

        return render_template(template, feature=detail, feature_type=feature_type)

    except Exception as e:
        safe_feature_type = feature_type.replace("\r", "").replace("\n", "")
        safe_feature_id = feature_id.replace("\r", "").replace("\n", "")
        logger.error(
            f"Error loading detail for {safe_feature_type} {safe_feature_id}: {e}",
            exc_info=True,
        )
        return '<div class="alert alert-error">Error loading feature details</div>'


@dashboard_bp.route("/export/configuration/<path:repo_path>")
def export_configuration(repo_path: str):
    """
    Export configuration features as JSON.

    Args:
        repo_path: Path to the repository

    Returns:
        JSON file download
    """
    from flask import jsonify, Response
    import json as json_lib

    try:
        service = current_app.dashboard_service
        # Flask's path converter strips leading slash, so add it back if missing
        if not repo_path.startswith("/"):
            repo_path = "/" + repo_path
        features = service.get_configuration_features(repo_path)

        # Find repository name
        repos = service.get_discovered_repositories()
        repo_name = "configuration"
        for repo in repos:
            if repo["path"] == repo_path:
                repo_name = repo["name"].replace(" ", "_").lower()
                break

        # Create JSON response
        json_data = json_lib.dumps(features, indent=2, default=str)

        return Response(
            json_data,
            mimetype="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={repo_name}_configuration.json"
            },
        )
    except Exception as e:
        logger.error(f"Error exporting configuration: {e}", exc_info=True)
        return jsonify({"error": "Failed to export configuration"}), 500


@dashboard_bp.route("/api/dashboard")
def api_dashboard() -> str:
    """
    API endpoint for HTMX to fetch filtered dashboard data.

    Query params:
        period: today|week|month|all (default: all)

    Returns:
        Rendered HTML partial with dashboard content
    """
    from flask import request

    try:
        service = current_app.dashboard_service
        period = request.args.get("period", "all")

        # Create time filter based on period
        time_filter = get_time_filter_from_period(period)

        # Fetch filtered data
        usage_summary: Dict[str, Any] = service.get_usage_summary(
            time_filter=time_filter
        )
        token_summary: Dict[str, Any] = service.get_token_summary(
            time_filter=time_filter
        )
        project_breakdown: Dict[str, Any] = service.get_project_breakdown(
            time_filter=time_filter
        )
        model_breakdown: Dict[str, Any] = service.get_model_breakdown(
            time_filter=time_filter
        )

        # For "All Time", show last 14 days on chart for better context
        chart_days = 14 if period == "all" else 7
        token_trend: Dict[str, Any] = service.get_daily_token_trend(
            days=chart_days, time_filter=time_filter
        )
        cost_trend: Dict[str, Any] = service.get_daily_cost_trend(
            days=chart_days, time_filter=time_filter
        )
        project_cost_trend: Dict[str, Any] = service.get_project_cost_trend(
            days=chart_days, time_filter=time_filter, max_projects=8
        )

        # Render partial template
        return render_template(
            "partials/dashboard_content.html",
            usage=usage_summary,
            tokens=token_summary,
            projects=project_breakdown,
            models=model_breakdown,
            token_trend=token_trend,
            cost_trend=cost_trend,
            project_cost_trend=project_cost_trend,
        )

    except Exception as e:
        logger.error(f"Error loading filtered dashboard: {e}", exc_info=True)
        return '<div class="text-red-600 p-4">Error loading data</div>', 500


@dashboard_bp.route("/api/projects")
def api_projects() -> str:
    """
    API endpoint for HTMX to fetch filtered projects data.

    Query params:
        period: today|week|month|all (default: all)
        model: model ID to filter by (default: all models)

    Returns:
        Rendered HTML partial with projects content
    """
    from flask import request

    try:
        service = current_app.dashboard_service
        period = request.args.get("period", "all")
        model = request.args.get("model", "")

        # Create time filter based on period
        time_filter = get_time_filter_from_period(period)

        # Fetch data - filtered by model if specified
        if model:
            project_breakdown = service.get_project_breakdown_by_model(
                model, time_filter=time_filter
            )
        else:
            project_breakdown = service.get_project_breakdown(time_filter=time_filter)

        available_models = service.get_available_models(time_filter=time_filter)

        # Render partial content
        return render_template(
            "partials/projects_content.html",
            projects=project_breakdown,
            available_models=available_models,
            selected_model=model,
        )

    except Exception as e:
        logger.error(f"Error loading filtered projects: {e}", exc_info=True)
        return (
            '<div class="text-red-600 p-4">Error loading projects</div>',
            500,
        )


@dashboard_bp.route("/api/files")
def api_files() -> str:
    """API endpoint for HTMX to fetch filtered files data."""
    from flask import request

    try:
        service = current_app.dashboard_service
        period = request.args.get("period", "all")

        time_filter = get_time_filter_from_period(period)

        file_data = service.get_file_statistics(limit=20, time_filter=time_filter)
        return render_template("partials/files_content.html", files=file_data)

    except Exception as e:
        logger.error(f"Error loading filtered files: {e}", exc_info=True)
        return '<div class="text-red-600 p-4">Error loading files</div>', 500


@dashboard_bp.route("/api/integrations")
def api_integrations() -> str:
    """API endpoint for HTMX to fetch filtered integrations data."""
    from flask import request

    try:
        service = current_app.dashboard_service
        period = request.args.get("period", "all")

        time_filter = get_time_filter_from_period(period)

        mcp_data = service.get_mcp_integrations(time_filter=time_filter)
        integrations_data = {
            "active_count": mcp_data["total_servers"],
            "total_calls": mcp_data["total_calls"],
            "total_tokens": mcp_data.get("total_tokens", 0),
            "total_cost": mcp_data.get("total_cost", 0.0),
            "server_activity": mcp_data["servers"],
        }
        return render_template(
            "partials/integrations_content.html", integrations=integrations_data
        )

    except Exception as e:
        logger.error(f"Error loading filtered integrations: {e}", exc_info=True)
        return (
            '<div class="text-red-600 p-4">Error loading integrations</div>',
            500,
        )


@dashboard_bp.route("/api/tokens")
def api_tokens() -> str:
    """API endpoint for HTMX to fetch filtered tokens data."""
    from flask import request

    try:
        service = current_app.dashboard_service
        period = request.args.get("period", "all")
        model = request.args.get("model", "")

        time_filter = get_time_filter_from_period(period)

        # Get token data - filtered by model if specified
        if model:
            token_summary = service.get_token_summary_by_model(
                model, time_filter=time_filter
            )
        else:
            token_summary = service.get_token_summary(time_filter=time_filter)

        model_breakdown = service.get_model_breakdown(time_filter=time_filter)
        available_models = service.get_available_models(time_filter=time_filter)

        return render_template(
            "partials/tokens_content.html",
            tokens=token_summary,
            models=model_breakdown,
            available_models=available_models,
            selected_model=model,
        )

    except Exception as e:
        logger.error(f"Error loading filtered tokens: {e}", exc_info=True)
        return (
            '<div class="text-red-600 p-4">Error loading tokens</div>',
            500,
        )


@dashboard_bp.route("/api/features")
def api_features() -> str:
    """API endpoint for HTMX to fetch filtered features data."""
    from flask import request

    try:
        service = current_app.dashboard_service
        period = request.args.get("period", "all")

        time_filter = get_time_filter_from_period(period)
        prev_time_filter = time_filter.get_previous_period() if time_filter else None

        # Get tool usage data for current period
        top_tools = service.get_top_tools(limit=20, time_filter=time_filter)
        features_summary = service.get_features_summary(time_filter=time_filter)

        # Get previous period data for trend comparison
        prev_tools_by_name = {}
        if prev_time_filter and prev_time_filter.start_time:
            prev_top_tools = service.get_top_tools(
                limit=50, time_filter=prev_time_filter
            )
            for tool in prev_top_tools.get("tools", []):
                prev_tools_by_name[tool.get("tool_name", "")] = tool

        # Build tool_usage list with proper field names for template
        tool_usage = []
        for tool in top_tools.get("tools", []):
            tool_name = tool.get("tool_name", "")
            cost_per_call = tool.get("cost_per_call", 0.0)

            # Calculate trend vs previous period
            trend = None
            if tool_name in prev_tools_by_name:
                prev_cost_per_call = prev_tools_by_name[tool_name].get(
                    "cost_per_call", 0
                )
                if prev_cost_per_call > 0:
                    change_percent = (
                        (cost_per_call - prev_cost_per_call) / prev_cost_per_call
                    ) * 100
                    trend = {
                        "change_percent": round(change_percent, 1),
                        "direction": "up"
                        if change_percent > 0
                        else "down"
                        if change_percent < 0
                        else "flat",
                        "prev_cost_per_call": prev_cost_per_call,
                    }

            tool_usage.append(
                {
                    "name": tool_name,
                    "category": tool.get("category", "tool"),
                    "usage_count": tool.get("invocation_count", 0),
                    "total_tokens": tool.get("total_tokens", 0),
                    "cost": tool.get("total_cost", 0.0),
                    "cost_per_call": cost_per_call,
                    "trend": trend,
                }
            )

        features_data = {
            "total_tools": len(tool_usage),
            "total_calls": top_tools.get("total_calls", 0),
            "total_tokens": top_tools.get("total_tokens", 0),
            "total_cost": top_tools.get("total_cost", 0.0),
            "sub_agents_used": features_summary.get("subagents", {}).get(
                "unique_used", 0
            ),
            "tool_usage": tool_usage,
            "categories": features_summary.get("categories", {}),
            "has_trends": prev_time_filter is not None
            and prev_time_filter.start_time is not None,
        }

        return render_template("partials/features_content.html", features=features_data)

    except Exception as e:
        logger.error(f"Error loading filtered features: {e}", exc_info=True)
        return (
            '<div class="text-red-600 p-4">Error loading features. Please try again later.</div>',
            500,
        )


@dashboard_bp.route("/api/project/analyze")
def api_project_analyze() -> Any:
    """
    API endpoint to analyze a project for optimization recommendations.

    Query params:
        project_path: Full path to the project
        project_name: Display name of the project
        period: today|week|month|all (default: all)

    Returns:
        JSON with project analysis and recommendations
    """
    from flask import request, jsonify
    from urllib.parse import unquote

    try:
        service = current_app.dashboard_service
        project_path = request.args.get("project_path", "")
        project_name = request.args.get("project_name", "Unknown")
        period = request.args.get("period", "all")

        # URL decode the project path
        if project_path:
            project_path = unquote(project_path)

        # Create time filter based on period
        time_filter = get_time_filter_from_period(period)

        # Get project analysis
        analysis = service.get_project_analysis(
            project_path=project_path,
            project_name=project_name,
            time_filter=time_filter,
        )

        return jsonify(analysis), 200

    except Exception as e:
        logger.error(f"Error analyzing project: {e}")
        return jsonify(
            {
                "error": "An error occurred while analyzing the project",
                "project_name": project_name,
                "recommendations": [],
                "metrics": {},
                "summary": {
                    "total": 0,
                    "high_severity": 0,
                    "medium_severity": 0,
                    "low_severity": 0,
                },
            }
        ), 500


@dashboard_bp.route("/settings")
def settings() -> str:
    """Render the settings page with pricing configuration."""
    try:
        service = current_app.dashboard_service

        # Dynamically load models from session data
        models_from_sessions = service.get_all_models_from_sessions()

        # Get pricing settings with dynamically discovered models
        pricing_settings = service.get_pricing_settings(
            additional_models=models_from_sessions
        )

        return render_template("pages/settings.html", pricing=pricing_settings)

    except Exception as e:
        logger.error(f"Error loading settings page: {e}", exc_info=True)
        return render_template(
            "pages/settings.html",
            pricing={
                "all_pricing": {},
                "custom_pricing": {},
                "has_custom_pricing": False,
            },
        )


@dashboard_bp.route("/export/<format>")
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

        if format not in ["csv", "json"]:
            return "Invalid format. Use 'csv' or 'json'", 400

        # Fetch all data
        usage_summary = service.get_usage_summary()
        token_summary = service.get_token_summary()
        project_breakdown = service.get_project_breakdown()

        if format == "json":
            # Export as JSON
            from flask import jsonify

            data = {
                "usage": usage_summary,
                "tokens": token_summary,
                "projects": project_breakdown,
                "exported_at": datetime.now().isoformat(),
            }
            response = make_response(jsonify(data))
            response.headers["Content-Disposition"] = (
                "attachment; filename=claudesavvy_export.json"
            )
            response.headers["Content-Type"] = "application/json"
            return response

        elif format == "csv":
            # Export as CSV
            output = io.StringIO()
            writer = csv.writer(output)

            # Write usage summary
            writer.writerow(["Usage Summary"])
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Total Commands", usage_summary.get("total_commands", 0)])
            writer.writerow(["Total Sessions", usage_summary.get("total_sessions", 0)])
            writer.writerow(["Total Messages", usage_summary.get("total_messages", 0)])
            writer.writerow(["Active Projects", usage_summary.get("total_projects", 0)])
            writer.writerow([])

            # Write token summary
            writer.writerow(["Token Summary"])
            writer.writerow(["Token Type", "Count", "Cost"])
            writer.writerow(
                [
                    "Input Tokens",
                    token_summary.get("input_tokens", 0),
                    f"${token_summary.get('input_cost', 0):.2f}",
                ]
            )
            writer.writerow(
                [
                    "Output Tokens",
                    token_summary.get("output_tokens", 0),
                    f"${token_summary.get('output_cost', 0):.2f}",
                ]
            )
            writer.writerow(
                [
                    "Cache Creation",
                    token_summary.get("cache_creation_tokens", 0),
                    f"${token_summary.get('cache_creation_cost', 0):.2f}",
                ]
            )
            writer.writerow(
                [
                    "Cache Reads",
                    token_summary.get("cache_read_tokens", 0),
                    f"${token_summary.get('cache_read_cost', 0):.2f}",
                ]
            )
            writer.writerow(
                ["Total Cost", "-", f"${token_summary.get('total_cost', 0):.2f}"]
            )
            writer.writerow([])

            # Write projects
            if project_breakdown.get("projects"):
                writer.writerow(["Projects"])
                writer.writerow(
                    ["Name", "Commands", "Sessions", "Messages", "Total Tokens", "Cost"]
                )
                for project in project_breakdown["projects"]:
                    writer.writerow(
                        [
                            project.get("name", ""),
                            project.get("command_count", 0),
                            project.get("session_count", 0),
                            project.get("message_count", 0),
                            project.get("total_tokens", 0),
                            f"${project.get('total_cost', 0):.2f}",
                        ]
                    )

            response = make_response(output.getvalue())
            response.headers["Content-Disposition"] = (
                "attachment; filename=claudesavvy_export.csv"
            )
            response.headers["Content-Type"] = "text/csv"
            return response

    except Exception as e:
        logger.error(f"Error exporting data: {e}", exc_info=True)
        return (
            "Error exporting data. Please check the server logs for more details.",
            500,
        )


@dashboard_bp.route("/api/settings/pricing")
def api_pricing_settings() -> Any:
    """
    API endpoint to get pricing settings.

    Returns:
        JSON with pricing settings for all models.
    """
    from flask import jsonify

    try:
        service = current_app.dashboard_service
        pricing_data = service.get_pricing_settings()
        return jsonify(pricing_data), 200

    except Exception as e:
        logger.error(f"Error getting pricing settings: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve pricing settings"}), 500


@dashboard_bp.route("/api/settings/pricing/update", methods=["POST"])
def api_update_pricing() -> Any:
    """
    API endpoint to update pricing for a specific model.

    Expects JSON body with:
        - model: Model identifier
        - input_per_mtok: Price per million input tokens
        - output_per_mtok: Price per million output tokens
        - cache_write_per_mtok: Price per million cache write tokens
        - cache_read_per_mtok: Price per million cache read tokens

    Returns:
        JSON with success status and updated pricing.
    """
    from flask import request, jsonify

    try:
        service = current_app.dashboard_service
        data = request.get_json()

        # Validate required fields
        required_fields = [
            "model",
            "input_per_mtok",
            "output_per_mtok",
            "cache_write_per_mtok",
            "cache_read_per_mtok",
        ]
        for field in required_fields:
            if field not in data:
                return jsonify(
                    {"success": False, "error": f"Missing field: {field}"}
                ), 400

        # Update pricing
        result = service.update_model_pricing(
            model=data["model"],
            input_per_mtok=float(data["input_per_mtok"]),
            output_per_mtok=float(data["output_per_mtok"]),
            cache_write_per_mtok=float(data["cache_write_per_mtok"]),
            cache_read_per_mtok=float(data["cache_read_per_mtok"]),
        )

        return jsonify(result), 200 if result["success"] else 500

    except ValueError:
        return jsonify({"success": False, "error": "Invalid value provided"}), 400
    except Exception as e:
        logger.error(f"Error updating pricing: {e}", exc_info=True)
        return jsonify({"success": False, "error": "Failed to update pricing"}), 500


@dashboard_bp.route("/api/settings/pricing/reset", methods=["POST"])
def api_reset_pricing() -> Any:
    """
    API endpoint to reset pricing for a specific model to default.

    Expects JSON body with:
        - model: Model identifier

    Returns:
        JSON with success status and default pricing.
    """
    from flask import request, jsonify

    try:
        service = current_app.dashboard_service
        data = request.get_json()

        if "model" not in data:
            return jsonify({"success": False, "error": "Missing field: model"}), 400

        # Reset pricing
        result = service.reset_model_pricing(data["model"])

        return jsonify(result), 200 if result["success"] else 500

    except Exception as e:
        logger.error(f"Error resetting pricing: {e}", exc_info=True)
        return jsonify({"success": False, "error": "Failed to reset pricing"}), 500


@dashboard_bp.route("/subagents")
def subagents() -> str:
    """Render the sub-agents page with exchange analytics."""
    try:
        service = current_app.dashboard_service
        # Default to 1hour filter on initial page load
        default_period = "1hour"
        time_filter = get_time_filter_from_period(default_period)

        # Get summary stats
        summary = service.get_subagent_summary(time_filter=time_filter)
        exchanges = service.get_subagent_exchanges(limit=50, time_filter=time_filter)
        chart_data = service.get_subagent_chart_data(limit=100, time_filter=time_filter)

        return render_template(
            "pages/subagents.html",
            summary=summary,
            exchanges=exchanges,
            chart_data=chart_data,
            period=default_period,
        )

    except Exception as e:
        logger.error(f"Error loading subagents page: {e}", exc_info=True)
        return render_template(
            "pages/subagents.html",
            summary={},
            exchanges={"exchanges": [], "total_count": 0},
            chart_data={"datasets": [], "total_points": 0},
        )


@dashboard_bp.route("/api/subagents")
def api_subagents() -> str:
    """API endpoint for HTMX to fetch filtered sub-agent data."""
    from flask import request

    try:
        service = current_app.dashboard_service
        period = request.args.get("period", "all")

        time_filter = get_time_filter_from_period(period)

        summary = service.get_subagent_summary(time_filter=time_filter)
        exchanges = service.get_subagent_exchanges(time_filter=time_filter, limit=50)
        chart_data = service.get_subagent_chart_data(time_filter=time_filter, limit=100)

        return render_template(
            "partials/subagents_content.html",
            summary=summary,
            exchanges=exchanges,
            chart_data=chart_data,
        )

    except Exception as e:
        logger.error(f"Error loading filtered subagents: {e}", exc_info=True)
        return '<div class="text-red-600 p-4">Error loading sub-agent data.</div>', 500


@dashboard_bp.route("/api/subagent/<agent_id>")
def api_subagent_detail(agent_id: str) -> str:
    """API endpoint for HTMX to fetch sub-agent exchange detail."""
    try:
        service = current_app.dashboard_service

        exchange = service.get_subagent_exchange_detail(agent_id)

        if not exchange:
            return '<div class="text-gray-500 p-4">Exchange not found.</div>', 404

        return render_template("partials/subagent_detail.html", exchange=exchange)

    except Exception as e:
        logger.error(f"Error loading subagent detail: {e}", exc_info=True)
        return (
            '<div class="text-red-600 p-4">Error loading exchange details.</div>',
            500,
        )
