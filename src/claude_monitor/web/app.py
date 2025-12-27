"""Flask application for Claude Monitor web interface.

This module provides the Flask app factory for creating and configuring
the Claude Monitor web application.
"""

import os
from datetime import datetime
from flask import Flask, jsonify
from typing import Optional

from ..utils.paths import ClaudeDataPaths, get_claude_paths
from .services.dashboard_service import DashboardService


def create_app(claude_data_paths: Optional[ClaudeDataPaths] = None) -> Flask:
    """
    Create and configure Flask application.

    Args:
        claude_data_paths: Optional ClaudeDataPaths instance. If not provided,
                         uses default Claude data directory.

    Returns:
        Configured Flask application
    """
    # Get the web module directory for template and static paths
    web_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(web_dir, 'templates')
    static_dir = os.path.join(web_dir, 'static')

    # Create Flask app with explicit template and static directories
    app = Flask(
        __name__,
        template_folder=template_dir,
        static_folder=static_dir,
        static_url_path='/static'
    )

    # Configuration
    app.config['JSON_SORT_KEYS'] = False

    # Jinja2 configuration
    app.jinja_env.auto_reload = True
    app.jinja_env.cache = None if app.debug else {}

    # Add template globals for use in templates
    app.jinja_env.globals.update(
        now=datetime.now,
        str=str,
        len=len,
        enumerate=enumerate,
    )

    # Add custom Jinja2 filters
    @app.template_filter('format_number')
    def format_number(value: float, decimals: int = 2) -> str:
        """Format a number with specified decimal places."""
        if value is None:
            return '0'
        try:
            return f'{float(value):.{decimals}f}'
        except (ValueError, TypeError):
            return str(value)

    @app.template_filter('format_percent')
    def format_percent(value: float, decimals: int = 1) -> str:
        """Format a value as a percentage."""
        if value is None:
            return '0%'
        try:
            return f'{float(value):.{decimals}f}%'
        except (ValueError, TypeError):
            return str(value)

    @app.template_filter('format_time')
    def format_time(seconds: float) -> str:
        """Format seconds into a human-readable time string."""
        if seconds is None or seconds < 0:
            return '0s'

        if seconds < 60:
            return f'{int(seconds)}s'
        elif seconds < 3600:
            minutes = seconds / 60
            return f'{minutes:.1f}m'
        else:
            hours = seconds / 3600
            return f'{hours:.1f}h'

    @app.template_filter('format_compact')
    def format_compact(value: float) -> str:
        """Format a large number in compact form (e.g., 1.5M, 234K)."""
        if value is None:
            return '0'
        try:
            num = float(value)
            if num >= 1_000_000_000:
                return f'{num / 1_000_000_000:,.2f}B'
            elif num >= 1_000_000:
                return f'{num / 1_000_000:,.2f}M'
            elif num >= 1_000:
                return f'{num / 1_000:,.1f}K'
            else:
                return f'{num:,.0f}'
        except (ValueError, TypeError):
            return str(value)

    # Initialize dashboard service
    if claude_data_paths is None:
        claude_data_paths = get_claude_paths()

    dashboard_service = DashboardService(claude_data_paths)

    # Store service in app context for routes to access
    app.dashboard_service = dashboard_service

    # Register blueprints (if they exist)
    # These will be imported from routes submodule
    try:
        from .routes.dashboard import dashboard_bp
        app.register_blueprint(dashboard_bp)
    except (ImportError, ModuleNotFoundError):
        # Dashboard blueprint is optional
        pass

    try:
        from .routes.api import api_bp
        app.register_blueprint(api_bp)
    except (ImportError, ModuleNotFoundError):
        # API blueprint is optional
        pass

    try:
        from .routes.charts import charts_bp
        app.register_blueprint(charts_bp)
    except (ImportError, ModuleNotFoundError):
        # Charts blueprint is optional
        pass

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 Not Found errors."""
        return jsonify({
            'error': 'Not Found',
            'message': 'The requested resource was not found.',
            'status': 404
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 Internal Server errors."""
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred. Please try again later.',
            'status': 500
        }), 500

    @app.errorhandler(400)
    def bad_request(error):
        """Handle 400 Bad Request errors."""
        return jsonify({
            'error': 'Bad Request',
            'message': 'The request could not be processed.',
            'status': 400
        }), 400

    # Add health check endpoint
    @app.route('/health')
    def health():
        """Health check endpoint."""
        return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()}), 200

    # Default route
    @app.route('/')
    def index():
        """Index route - display welcome message."""
        return jsonify({
            'message': 'Claude Monitor Web Server',
            'version': '1.0.0',
            'status': 'running',
            'timestamp': datetime.now().isoformat()
        }), 200

    # Add CORS headers for development (if needed)
    @app.after_request
    def add_headers(response):
        """Add headers to response for development."""
        # Allow same-origin requests by default
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        response.headers['X-XSS-Protection'] = '1; mode=block'

        # In development, add CORS headers
        if app.debug:
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'

        return response

    return app
