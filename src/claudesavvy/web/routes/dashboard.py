"""Dashboard route handler for Claude Monitor web application."""

import importlib.resources
import json
import re
import subprocess
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional
from flask import Blueprint, render_template, current_app
import logging

from ...utils.paths import ClaudeDataPaths
from ...utils.time_filter import TimeFilter

logger = logging.getLogger(__name__)

dashboard_bp = Blueprint("dashboard", __name__)

# Valid period preset names for validation
_VALID_PERIODS = frozenset({
    "15min", "today", "this_week", "7days", "this_month", "3months", "all",
    # legacy aliases kept for bookmarked URLs
    "week", "month", "1hour",
})


def get_time_filter_from_period(
    period: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Optional[TimeFilter]:
    """
    Create a TimeFilter from a period string.

    Args:
        period: One of '15min', 'today', 'this_week', '7days', 'this_month',
                '3months', 'all', or 'custom'
        start: ISO date string (YYYY-MM-DD) used when period='custom'
        end: ISO date string (YYYY-MM-DD) used when period='custom'

    Returns:
        TimeFilter instance or None for 'all'
    """
    if period == "all":
        return None

    # Custom date range
    if period == "custom" and start and end:
        try:
            start_date = date.fromisoformat(start)
            end_date = date.fromisoformat(end)
            return TimeFilter.from_range(start_date, end_date)
        except (ValueError, TypeError):
            period = "today"

    # Named presets (including legacy aliases for bookmarked URLs)
    preset_map = {
        "15min": "15min",
        "today": "today",
        "this_week": "this_week",
        "7days": "7days",
        "this_month": "this_month",
        "3months": "3months",
        "week": "7days",
        "month": "this_month",
        "1hour": "15min",  # closest supported preset to the old 1-hour window
    }

    preset = preset_map.get(period)
    if preset:
        return TimeFilter.from_preset(preset)

    # Unknown period: fall back to "today" rather than silently returning all-time
    if period not in _VALID_PERIODS:
        logger.warning("Unknown period %r requested, defaulting to 'today'", period)
    return TimeFilter.from_preset("today")


def _build_preview_data(service: Any, time_filter: Optional[TimeFilter]) -> Dict[str, Any]:
    """Build preview data for the dashboard navigation hub cards."""
    conv_analytics: Dict[str, Any] = service.get_conversation_analytics(
        time_filter=time_filter, limit=200
    )
    convs = conv_analytics.get("conversations", [])
    recent_conversations = sorted(
        convs,
        key=lambda x: x.get("start_time") or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )[:5]
    conversations_preview: Dict[str, Any] = {
        "recent": recent_conversations,
        "total_count": conv_analytics.get("summary", {}).get("total_conversations", 0),
        "most_expensive_cost": conv_analytics.get("summary", {}).get("most_expensive_cost", 0),
        "most_expensive_project": convs[0].get("project_name", "") if convs else "",
        "most_expensive_session_id": conv_analytics.get("summary", {}).get("most_expensive_session_id", ""),
    }

    subagents_preview: Dict[str, Any] = service.get_subagent_summary(
        time_filter=time_filter
    )

    file_data: Dict[str, Any] = service.get_file_statistics(
        limit=3, time_filter=time_filter
    )
    files_preview: Dict[str, Any] = {
        "total_files": file_data.get("total_files", 0),
        "most_edited_file": file_data.get("most_edited_file", ""),
        "total_operations": file_data.get("total_operations", 0),
    }

    # Use the same transcript-derived source as the /integrations page so the
    # preview and the page agree on whether MCP servers were active.
    mcp_data: Dict[str, Any] = service.get_mcp_integrations(time_filter=time_filter)
    integrations_preview: Dict[str, Any] = {
        "has_integrations": mcp_data.get("total_servers", 0) > 0,
        "total_servers": mcp_data.get("total_servers", 0),
        "total_tool_calls": mcp_data.get("total_calls", 0),
        "top_servers": [
            {"server_name": s["server_name"], "tool_call_count": s["total_calls"]}
            for s in mcp_data.get("servers", [])[:1]
        ],
    }

    top_tools: Dict[str, Any] = service.get_top_tools(limit=1, time_filter=time_filter)
    features_preview: Dict[str, Any] = {
        "top_tool": top_tools["tools"][0] if top_tools.get("tools") else None,
        "total_tool_calls": top_tools.get("total_calls", 0),
    }

    return {
        "conversations_preview": conversations_preview,
        "subagents_preview": subagents_preview,
        "files_preview": files_preview,
        "integrations_preview": integrations_preview,
        "features_preview": features_preview,
    }


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

        # Default to today for a meaningful overview
        time_filter = get_time_filter_from_period("today")

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
        cost_trend: Dict[str, Any] = service.get_daily_cost_trend(
            time_filter=time_filter
        )
        project_cost_trend: Dict[str, Any] = service.get_project_cost_trend(
            max_projects=8, time_filter=time_filter
        )

        # Preview data for navigation hub cards
        previews = _build_preview_data(service, time_filter)

        logger.debug("Dashboard data fetched successfully")

        # Render template with all data
        return render_template(
            "pages/dashboard.html",
            usage=usage_summary,
            tokens=token_summary,
            projects=project_breakdown,
            models=model_breakdown,
            cost_trend=cost_trend,
            project_cost_trend=project_cost_trend,
            period="today",
            **previews,
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
        default_period = "today"
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
        default_period = "today"
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
        default_period = "today"
        time_filter = get_time_filter_from_period(default_period)

        # Get file statistics
        file_data = service.get_file_statistics(limit=100, time_filter=time_filter)

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
        default_period = "today"
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
        default_period = "today"
        time_filter = get_time_filter_from_period(default_period)

        # Get tool usage data
        top_tools = service.get_top_tools(limit=100, time_filter=time_filter)
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


def _annotate_feature_usage(features: Dict[str, Any], usage: Dict[str, Any]) -> None:
    """Stamp each skill/agent/MCP config item with its 30-day call count.

    Runtime names can be namespaced (e.g. 'plugin:skill'), so fall back to
    matching on the last ':'-separated segment.
    """
    def _match(name: str, counter: Dict[str, int]) -> int:
        if name in counter:
            return counter[name]
        short = name.split(":")[-1]
        return sum(
            n for used, n in counter.items() if used.split(":")[-1] == short
        )

    for key in ("skills", "agents", "mcps"):
        counter = usage.get(key, {})
        for item in features.get(key) or []:
            if isinstance(item, dict):
                item["used_count"] = _match(item.get("name", ""), counter)


def _project_scoped(features):
    """Return a copy of features containing only project-sourced items.

    User/plugin items are attached to every repo by the scanner but should
    only be counted once (via user_features).  This helper strips them out
    so project repos contribute only their own PROJECT-tagged items to totals.
    Non-list values (e.g. the 'counts' key) are dropped; the template only
    iterates the list keys.
    """
    return {
        key: [i for i in items if isinstance(i, dict) and i.get("source") == "project"]
        for key, items in features.items()
        if isinstance(items, list)
    }


@dashboard_bp.route("/harness")
def harness() -> str:
    """Render the Agent Harness overview page."""
    try:
        service = current_app.dashboard_service
        repositories = service.get_discovered_repositories()

        user_repo = None
        user_features = {}
        project_repos = []

        for repo in repositories:
            features = service.get_configuration_features(repo["path"])
            repo_with_features = dict(repo)
            repo_with_features["features"] = _project_scoped(features)

            if repo.get("type") == "user" or repo.get("name", "").lower() in ("user", "user configuration", "~/.claude"):
                user_repo = repo
                user_features = features
            else:
                project_repos.append(repo_with_features)

        if not user_repo and repositories:
            user_repo = repositories[0]
            user_features = service.get_configuration_features(repositories[0]["path"])
            project_repos = []
            for repo in repositories[1:]:
                features = service.get_configuration_features(repo["path"])
                repo_with_features = dict(repo)
                repo_with_features["features"] = _project_scoped(features)
                project_repos.append(repo_with_features)

        def _count(features, key):
            items = features.get(key) or []
            return len(items) if isinstance(items, list) else 0

        all_features_list = [user_features] + [r["features"] for r in project_repos if r.get("features")]

        harness_totals = {
            "skills": sum(_count(f, "skills") for f in all_features_list),
            "agents": sum(_count(f, "agents") for f in all_features_list),
            "mcps": sum(_count(f, "mcps") for f in all_features_list),
            "plugins": sum(_count(f, "plugins") for f in all_features_list),
            "hooks": sum(_count(f, "hooks") for f in all_features_list),
            "commands": sum(_count(f, "commands") for f in all_features_list),
        }

        # 30-day usage so the inventory shows dead weight vs. earning items
        try:
            usage = service.get_harness_usage(window_days=30)
        except Exception:
            logger.warning("Could not compute harness usage", exc_info=True)
            usage = {"skills": {}, "agents": {}, "mcps": {}, "window_days": 30}
        for f in all_features_list:
            _annotate_feature_usage(f, usage)

        def _used(key):
            return sum(
                1
                for f in all_features_list
                for item in (f.get(key) or [])
                if isinstance(item, dict) and item.get("used_count")
            )

        harness_used = {
            "skills": _used("skills"),
            "agents": _used("agents"),
            "mcps": _used("mcps"),
        }

        default_period = "7days"
        time_filter = get_time_filter_from_period(default_period)
        evaluation = service.get_harness_evaluation(time_filter=time_filter)
        harness_projects = service.get_harness_projects()

        return render_template(
            "pages/harness.html",
            repositories=repositories,
            user_repo=user_repo,
            user_features=user_features,
            project_repos=project_repos,
            harness_totals=harness_totals,
            harness_used=harness_used,
            evaluation=evaluation,
            period=default_period,
            harness_projects=harness_projects,
        )

    except Exception as e:
        logger.error(f"Error loading harness page: {e}", exc_info=True)
        return render_template(
            "pages/harness.html",
            repositories=[],
            user_repo=None,
            user_features={},
            project_repos=[],
            harness_totals={"skills": 0, "agents": 0, "mcps": 0, "plugins": 0, "hooks": 0, "commands": 0},
            harness_used={},
            evaluation=None,
            period="7days",
            harness_projects=[],
        )


@dashboard_bp.route("/api/harness/evaluation")
def api_harness_evaluation() -> str:
    """HTMX endpoint: filtered harness evaluation leaderboard."""
    from flask import request

    try:
        service = current_app.dashboard_service
        period = request.args.get("period", "7days")
        project = request.args.get("project") or None
        time_filter = get_time_filter_from_period(
            period, start=request.args.get("start"), end=request.args.get("end")
        )
        evaluation = service.get_harness_evaluation(time_filter=time_filter, project=project)
        return render_template(
            "partials/harness_evaluation.html",
            evaluation=evaluation,
            period=period,
        )
    except Exception as e:
        logger.error(f"Error loading harness evaluation: {e}", exc_info=True)
        return (
            '<div class="text-red-600 p-4">Error loading evaluation</div>',
            500,
        )


@dashboard_bp.route("/api/harness/waste-card")
def api_harness_waste_card() -> str:
    """HTMX endpoint: lazy-loaded dashboard callout for estimated wasted spend."""
    from flask import request

    try:
        service = current_app.dashboard_service
        period = request.args.get("period", "today")
        time_filter = get_time_filter_from_period(
            period, start=request.args.get("start"), end=request.args.get("end")
        )
        evaluation = service.get_harness_evaluation(time_filter=time_filter)
        return render_template(
            "partials/harness_waste_card.html",
            evaluation=evaluation,
        )
    except Exception as e:
        logger.error(f"Error loading waste card: {e}", exc_info=True)
        # Non-essential callout: fail silent rather than break the dashboard
        return ""


@dashboard_bp.route("/api/harness/session/<session_id>")
def api_harness_session(session_id: str) -> str:
    """HTMX endpoint: detail for a single scored session."""
    try:
        service = current_app.dashboard_service
        detail = service.get_harness_session_detail(session_id)
        if detail is None:
            return (
                '<div class="text-slate-400 p-4">Session not found</div>',
                404,
            )
        return render_template(
            "partials/harness_session_detail.html",
            session=detail,
        )
    except Exception as e:
        logger.error(f"Error loading harness session detail: {e}", exc_info=True)
        return (
            '<div class="text-red-600 p-4">Error loading session</div>',
            500,
        )


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
        time_filter = get_time_filter_from_period(period, start=request.args.get("start"), end=request.args.get("end"))

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

        cost_trend: Dict[str, Any] = service.get_daily_cost_trend(
            time_filter=time_filter
        )
        project_cost_trend: Dict[str, Any] = service.get_project_cost_trend(
            time_filter=time_filter, max_projects=8
        )

        previews = _build_preview_data(service, time_filter)

        # Render partial template
        return render_template(
            "partials/dashboard_content.html",
            usage=usage_summary,
            tokens=token_summary,
            projects=project_breakdown,
            models=model_breakdown,
            cost_trend=cost_trend,
            project_cost_trend=project_cost_trend,
            period=period,
            **previews,
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
        time_filter = get_time_filter_from_period(period, start=request.args.get("start"), end=request.args.get("end"))

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

        time_filter = get_time_filter_from_period(period, start=request.args.get("start"), end=request.args.get("end"))

        file_data = service.get_file_statistics(limit=100, time_filter=time_filter)
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

        time_filter = get_time_filter_from_period(period, start=request.args.get("start"), end=request.args.get("end"))

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

        time_filter = get_time_filter_from_period(period, start=request.args.get("start"), end=request.args.get("end"))

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

        time_filter = get_time_filter_from_period(period, start=request.args.get("start"), end=request.args.get("end"))
        prev_time_filter = time_filter.get_previous_period() if time_filter else None

        # Get tool usage data for current period
        top_tools = service.get_top_tools(limit=100, time_filter=time_filter)
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

        return render_template(
            "partials/features_content.html", features=features_data, period=period
        )

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
        time_filter = get_time_filter_from_period(period, start=request.args.get("start"), end=request.args.get("end"))

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


@dashboard_bp.route("/api/settings/pricing/sync", methods=["POST"])
def api_sync_pricing() -> Any:
    """
    API endpoint to sync model pricing from Anthropic's published
    pricing page. Synced prices become the new defaults; custom
    per-model overrides still take precedence.

    Returns:
        JSON with success status, synced pricing, and sync timestamp.
    """
    from flask import jsonify

    try:
        service = current_app.dashboard_service
        result = service.sync_pricing_from_web()
        return jsonify(result), 200 if result["success"] else 502

    except Exception as e:
        logger.error(f"Error syncing pricing: {e}", exc_info=True)
        return jsonify({"success": False, "error": "Failed to sync pricing"}), 500


@dashboard_bp.route("/subagents")
def subagents() -> str:
    """Render the sub-agents page with exchange analytics."""
    try:
        service = current_app.dashboard_service
        # Default to 1hour filter on initial page load
        default_period = "today"
        time_filter = get_time_filter_from_period(default_period)

        # Get summary stats
        summary = service.get_subagent_summary(time_filter=time_filter)
        exchanges = service.get_subagent_exchanges(limit=200, time_filter=time_filter)
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

        time_filter = get_time_filter_from_period(period, start=request.args.get("start"), end=request.args.get("end"))

        summary = service.get_subagent_summary(time_filter=time_filter)
        exchanges = service.get_subagent_exchanges(time_filter=time_filter, limit=200)
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


@dashboard_bp.route("/api/tool/<path:tool_name>")
def api_tool_detail(tool_name: str) -> str:
    """API endpoint for HTMX to fetch tool invocation details."""
    from flask import request

    try:
        service = current_app.dashboard_service
        period = request.args.get("period", "all")

        time_filter = get_time_filter_from_period(period, start=request.args.get("start"), end=request.args.get("end"))

        tool_data = service.get_tool_invocations(
            tool_name=tool_name,
            time_filter=time_filter,
            limit=100,
        )

        if not tool_data or tool_data.get("total_invocations", 0) == 0:
            return (
                '<div class="text-gray-500 p-4">No invocations found for this tool.</div>',
                404,
            )

        # Get chart data for timeline
        chart_data = service.get_tool_chart_data(
            tool_name=tool_name,
            time_filter=time_filter,
            limit=100,
        )

        return render_template(
            "partials/tool_detail.html",
            tool=tool_data,
            chart_data=chart_data,
            period=period,
        )

    except Exception as e:
        logger.error(f"Error loading tool detail: {e}", exc_info=True)
        return (
            '<div class="text-red-600 p-4">Error loading tool details.</div>',
            500,
        )


@dashboard_bp.route("/api/tool/<path:tool_name>/invocation/<path:invocation_id>")
def api_tool_invocation_detail(tool_name: str, invocation_id: str) -> str:
    """API endpoint for HTMX to fetch individual tool invocation details."""
    from flask import request

    try:
        service = current_app.dashboard_service
        period = request.args.get("period", "all")

        time_filter = get_time_filter_from_period(period, start=request.args.get("start"), end=request.args.get("end"))

        invocation = service.get_tool_invocation_detail(
            tool_name=tool_name,
            invocation_id=invocation_id,
            time_filter=time_filter,
        )

        if not invocation:
            return (
                '<div class="text-gray-500 p-4">Invocation not found.</div>',
                404,
            )

        return render_template(
            "partials/tool_invocation_detail.html",
            invocation=invocation,
        )

    except Exception as e:
        logger.error(f"Error loading tool invocation detail: {e}", exc_info=True)
        return (
            '<div class="text-red-600 p-4">Error loading invocation details.</div>',
            500,
        )


@dashboard_bp.route("/conversations")
def conversations() -> str:
    """Render the conversations analytics page."""
    try:
        service = current_app.dashboard_service
        default_period = "week"
        time_filter = get_time_filter_from_period(default_period)
        data = service.get_conversation_analytics(time_filter=time_filter)
        return render_template(
            "pages/conversations.html",
            period=default_period,
            **data,
        )
    except Exception as e:
        logger.error(f"Error loading conversations page: {e}", exc_info=True)
        return render_template(
            "pages/conversations.html",
            period="week",
            conversations=[],
            summary={},
        )


@dashboard_bp.route("/api/conversations")
def api_conversations() -> str:
    """HTMX endpoint to refresh conversations content."""
    from flask import request

    try:
        service = current_app.dashboard_service
        period = request.args.get("period", "week")
        time_filter = get_time_filter_from_period(period, start=request.args.get("start"), end=request.args.get("end"))
        data = service.get_conversation_analytics(time_filter=time_filter)
        return render_template("partials/conversations_content.html", period=period, **data)
    except Exception as e:
        logger.error(f"Error loading conversations data: {e}", exc_info=True)
        return '<div class="text-red-600 p-4">Error loading conversation data.</div>', 500


@dashboard_bp.route("/api/conversation/<session_id>")
def api_conversation_detail(session_id: str) -> str:
    """HTMX endpoint to load conversation detail panel."""
    from flask import request

    try:
        service = current_app.dashboard_service
        period = request.args.get("period", "week")
        time_filter = get_time_filter_from_period(period, start=request.args.get("start"), end=request.args.get("end"))
        detail = service.get_conversation_detail(session_id, time_filter=time_filter)
        if not detail:
            return '<div class="text-gray-500 p-4">Conversation not found.</div>', 404
        return render_template("partials/conversation_detail.html", conv=detail, period=period)
    except Exception as e:
        logger.error(f"Error loading conversation detail: {e}", exc_info=True)
        return '<div class="text-red-600 p-4">Error loading conversation details.</div>', 500


@dashboard_bp.route("/api/tools/timeline")
def api_tools_timeline() -> str:
    """API endpoint for HTMX to fetch tools timeline data."""
    from flask import request

    try:
        service = current_app.dashboard_service

        period = request.args.get("period", "all")
        session_id = request.args.get("session_id", "")

        time_filter = get_time_filter_from_period(period, start=request.args.get("start"), end=request.args.get("end"))

        timeline_data = service.get_unified_timeline_data(
            time_filter=time_filter,
            session_id=session_id if session_id else None,
        )

        return render_template(
            "partials/unified_tools_timeline.html",
            timeline=timeline_data,
            period=period,
        )

    except Exception as e:
        logger.error(f"Error loading tools timeline: {e}", exc_info=True)
        return (
            '<div class="text-red-600 p-4">Error loading timeline.</div>',
            500,
        )


@dashboard_bp.route("/teams")
def teams() -> str:
    """Render the teams page with team usage analytics."""
    try:
        service = current_app.dashboard_service
        # Default to all time for teams (typically fewer events)
        default_period = "all"
        time_filter = get_time_filter_from_period(default_period)

        # Get team summary stats
        summary = service.get_team_summary(time_filter=time_filter)
        chart_data = service.get_team_chart_data(time_filter=time_filter)

        return render_template(
            "pages/teams.html",
            summary=summary,
            chart_data=chart_data,
            period=default_period,
        )

    except Exception as e:
        logger.error(f"Error loading teams page: {e}", exc_info=True)
        return render_template(
            "pages/teams.html",
            summary={"teams": [], "total_teams": 0, "total_cost": 0, "total_tokens": 0},
            chart_data={"labels": [], "datasets": []},
            period="all",
        )


@dashboard_bp.route("/api/teams")
def api_teams() -> str:
    """API endpoint for HTMX to fetch filtered team data."""
    from flask import request

    try:
        service = current_app.dashboard_service
        period = request.args.get("period", "all")

        time_filter = get_time_filter_from_period(period, start=request.args.get("start"), end=request.args.get("end"))

        summary = service.get_team_summary(time_filter=time_filter)
        chart_data = service.get_team_chart_data(time_filter=time_filter)

        return render_template(
            "partials/teams_content.html",
            summary=summary,
            chart_data=chart_data,
        )

    except Exception as e:
        logger.error(f"Error loading filtered teams: {e}", exc_info=True)
        return '<div class="text-red-600 p-4">Error loading team data.</div>', 500


# ---------------------------------------------------------------------------
# Statusline installer
# ---------------------------------------------------------------------------

def _statusline_state() -> dict:
    """Return current install state and live preview output."""
    paths = ClaudeDataPaths()
    script_dest = paths.base_dir / "statusline.py"

    script_installed = script_dest.exists()

    settings_configured = False
    try:
        with open(paths.settings_file, encoding="utf-8") as f:
            cfg = json.load(f)
        sl = cfg.get("statusLine", {})
        settings_configured = sl.get("type") == "command" and "statusline.py" in sl.get("command", "")
    except Exception:
        # settings.json may not exist or be unreadable; treat as unconfigured
        pass

    preview = None
    if script_installed:
        try:
            result = subprocess.run(
                ["python3", str(script_dest)],
                capture_output=True, text=True, timeout=5
            )
            # Strip ANSI escape codes so the preview renders cleanly in HTML
            ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
            raw = result.stdout or result.stderr or "(no output)"
            preview = ansi_escape.sub("", raw)
        except Exception:
            preview = "(preview unavailable)"

    return {
        "installed": script_installed and settings_configured,
        "script_installed": script_installed,
        "settings_configured": settings_configured,
        "script_dest": str(script_dest),
        "preview": preview,
    }


@dashboard_bp.route("/api/statusline/status")
def api_statusline_status() -> str:
    """Return current statusline install state as HTML partial."""
    try:
        state = _statusline_state()
        return render_template("partials/statusline_card.html", **state)
    except Exception as e:
        logger.error("Error checking statusline status: %s", str(e), exc_info=True)
        return '<div class="text-red-600 p-4">Error checking statusline status.</div>', 500


@dashboard_bp.route("/api/statusline/install", methods=["POST"])
def api_statusline_install() -> str:
    """Install the statusline script and configure ~/.claude/settings.json."""
    from flask import request

    # Restrict to localhost — this endpoint writes to the user's ~/.claude/settings.json
    if request.remote_addr not in ("127.0.0.1", "::1"):
        return '<div class="text-red-600 p-4">Install only permitted from localhost.</div>', 403

    try:
        paths = ClaudeDataPaths()
        paths.base_dir.mkdir(parents=True, exist_ok=True)
        script_dest = paths.base_dir / "statusline.py"

        pkg_data = importlib.resources.files("claudesavvy").joinpath("data/statusline.py")
        script_dest.write_text(pkg_data.read_text(), encoding="utf-8")

        cfg: dict = {}
        try:
            with open(paths.settings_file, encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            # settings.json may not exist yet; start with empty config
            cfg = {}

        cfg["statusLine"] = {
            "type": "command",
            "command": f"python3 {script_dest}",
        }
        with open(paths.settings_file, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        state = _statusline_state()
        state["flash"] = "Installed! Restart Claude Code to activate."
        return render_template("partials/statusline_card.html", **state)

    except Exception as e:
        logger.error("Error installing statusline: %s", str(e), exc_info=True)
        return '<div class="text-red-600 p-4">Install failed. Check server logs for details.</div>', 500
