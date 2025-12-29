# Changelog

All notable changes to ClaudeSavvy will be documented in this file.

## [2.0.1] - 2025-12-29

### Fixed
- Fixed `AttributeError: 'list' object has no attribute 'get'` error in skills parser
- Added robust error handling for `installed_plugins.json` parsing
- Added type checking and logging for better debugging of plugin data structures
- Features page now correctly handles plugin entries that are organized as lists

### Changed
- Improved defensive programming in SkillsParser with proper type validation
- Added warning logs for unexpected data structures in plugin files

## [2.0.0] - 2025-12-29

### Changed
- **BREAKING**: Major rebrand from "Claude Monitor" to "ClaudeSavvy"
- **BREAKING**: Package name changed from `claude-monitor` to `claudesavvy` on PyPI
- **BREAKING**: Command changed from `claude-monitor` to `claudesavvy`
- Repository name remains as `claude_monitor` for historical reasons
- Updated all documentation to reflect new brand name
- Refreshed README with new branding, tagline, and visual identity
- Updated package description to emphasize "dashboard" functionality

### Fixed
- Reorganized dashboard page layout for better navigation (see #18)
- Updated all references in documentation and configuration files

### Migration Guide

If you were using Claude Monitor v1.x:

**Before (v1.x):**
```bash
pip install claude-monitor
claude-monitor
```

**Now (v2.0.0):**
```bash
pip install claudesavvy
claudesavvy
```

To upgrade from v1.x to v2.0.0:
```bash
pip uninstall claude-monitor
pip install claudesavvy
```

Note: Your data files remain unchanged. Only the package name and command have changed.

## [1.0.1] - 2025-12-27

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2025-12-27

### Added
- Automated PyPI publishing via GitHub Actions
- Cross-platform binary builds (macOS, Linux, Windows) using PyInstaller
- Pre-built executable distribution through GitHub Releases
- Complete release infrastructure and documentation

### Changed
- Installation now supports three methods: PyPI, pre-built binaries, and from source
- Updated README with installation instructions for all distribution methods

### Fixed
- Security: Fixed 8 instances of empty except blocks with specific exception handling
- Cross-platform compatibility in PyInstaller spec file
- Improved dependency handling for web components (Flask templates and static assets)

## [1.0.0] - 2025-12-20

### Added
- Web-based dashboard interface with Flask
- Multiple dedicated metric pages:
  - Dashboard: Overview of all metrics
  - Tokens: Detailed token usage and cost breakdown
  - Projects: Per-project analytics
  - Files: File operation statistics
  - Integrations: MCP server usage
  - Features: Tool usage and sub-agent statistics
- Data export functionality (CSV and JSON formats)
- Health check endpoint for monitoring
- HTMX-powered dynamic content updates
- Time period filtering (today, week, month, all-time)
- Responsive web UI with modern styling
- Production-ready documentation (LICENSE, CONTRIBUTING, CHANGELOG)

### Changed
- **BREAKING**: Removed CLI dashboard interface entirely
- **BREAKING**: `claude-monitor` command now launches web server directly (no subcommand required)
- Updated entry point from dual-mode (CLI/web) to web-only
- Standardized dependency versions across all configuration files
- Updated project description to "Web-based usage monitoring tool for Claude Code"
- Bumped version to 1.0.0 to reflect major interface change

### Removed
- **BREAKING**: CLI interactive menu and dashboard views
- **BREAKING**: CLI-specific options: `--today`, `--week`, `--month`, `--quarter`, `--year`, `--focus`, `--interactive`
- Entire `display/` module (dashboard, menu, tables, cards, formatter)
- CLI time filtering and project filtering flags

### Fixed
- Added Flask>=3.0.0 and Jinja2>=3.1.0 to setup.py (previously missing)
- Standardized Flask version to 3.0.0 across setup.py, pyproject.toml, and requirements.txt
- Fixed installation via `pip install .` which previously failed to install web dependencies

### Migration Guide

If you were using the CLI dashboard (v0.1.0):

**Before (v0.1.0):**
```bash
claude-monitor                    # CLI dashboard
claude-monitor --week             # CLI with time filter
claude-monitor web                # Web server
```

**Now (v1.0.0):**
```bash
claude-monitor                    # Web server (default)
claude-monitor --port 8080        # Web server on custom port
claude-monitor --debug            # Web server in debug mode
```

To continue using the CLI dashboard, stay on version 0.1.0:
```bash
git checkout v0.1.0
pip install -e .
```

## [0.1.0] - 2025-12-19

### Added
- Initial release with CLI dashboard interface
- Interactive menu with logo and navigation
- Usage metrics tracking (sessions, commands, projects)
- Token usage and cost calculations
- Prompt cache efficiency monitoring
- Project breakdown analytics
- File modification tracking
- MCP integration statistics
- Model usage breakdown
- Time-based filtering (today, week, month, quarter, year, custom dates)
- Rich terminal UI with tables and color-coded output
- Support for custom Claude data directory
- Web interface as optional subcommand

### Data Sources
- Reads from `~/.claude/` directory
- Parses history.jsonl for command history
- Analyzes session files for token usage
- Processes debug logs for MCP activity
- Tracks file editing history

### Requirements
- Python 3.9+
- click>=8.1.0
- rich>=13.0.0
- python-dateutil>=2.8.0
- Flask>=2.3.0 (for web mode)

---

## Version History

- **1.0.0** (2025-12-20): Web-only release with production polish
- **0.1.0** (2025-12-19): Initial release with dual CLI/web interface
