# Changelog

All notable changes to ClaudeSavvy will be documented in this file.

## [3.0.0] - 2026-06-12

### Added
- **New**: Settings → Pricing now lists every built-in model up front (models found in your session data are badged "In use"), so applicable rates are visible before a model is ever used
- **New**: "Sync prices from anthropic.com" button on the Settings page — fetches the published pricing page and stores current rates locally; synced prices become the new defaults while custom per-model overrides still take precedence
- **New**: Built-in model pricing table synced with official Claude API rates, including Claude Fable 5, Opus 4.8/4.7/4.6/4.5, Opus 4.1/4 legacy rates, Sonnet 4.6, and Haiku 3.5/3
- **New**: Model pricing resolution handles dated IDs, Bedrock/Vertex prefixes, the `[1m]` context-window marker, and falls back by model family for future releases
- **New**: Client-side search, column sorting, and pagination on all major data tables (projects, conversations, tokens, files, features, integrations, sub-agents, teams)

### Changed
- Session file parsing is now cached per file (invalidated on mtime/size change) — steady-state page loads no longer re-parse unchanged JSONL files (~8x faster warm aggregation)
- Daily and per-project cost trends now use per-model pricing instead of flat Sonnet rates
- Table row caps raised now that tables paginate (files/tools 20 → 100, sub-agent exchanges 50 → 200)

### Fixed
- **Critical**: Models without a manual pricing override were silently billed at Sonnet rates ($3/$15), under-costing Opus ~5x and over-costing Haiku ~3x
- Statusline "today" was actually a rolling last-24-hours window, so yesterday's usage kept showing as today's; it now uses the local calendar day and month boundaries, matching the dashboard's "Today" and "This month" filters

### Removed
- GitHub Issue Agent (CLI command, module, workflow, and docs) — the feature only ever selected an issue and never implemented fixes or PR creation

## [2.9.0] - 2026-03-08

### Added
- **New**: Claude Code statusline with one-click installer in Settings
- `statusline.py`: standalone script showing today's cost, token count, cache hit %, session count, and month-to-date cost with ANSI colours
- Bundled as package data so it ships with the package and can be installed from any machine running the app
- `GET /api/statusline/status`: checks install state and runs a live preview
- `POST /api/statusline/install`: writes the script and patches `settings.json` to register the hook
- Settings page: new card with status badges, terminal preview, and Install/Reinstall button loaded via HTMX

### Security
- Restricted `POST /api/statusline/install` to localhost only (127.0.0.1/::1) to prevent remote config modification when server binds to 0.0.0.0
- ANSI escape codes stripped from preview output before rendering in HTML

## [2.8.0] - 2026-03-01

### Added
- **New**: Complete time filtering overhaul with new presets: this_week, 7days, this_month, 3months
- **New**: Custom date range support via period=custom&start=YYYY-MM-DD&end=YYYY-MM-DD on all API endpoints
- **New**: Shared time_filter.html partial component across all pages
- **New**: Per-page localStorage persistence for selected time period
- **New**: Dashboard redesigned as navigation hub with full page coverage
- **New**: Cost Over Time chart replacing Token Usage Over Time
- **New**: Recent Conversations mini-feed with context utilization bars
- **New**: 'Explore Your Usage' navigation grid covering all 6 sub-pages
- **New**: Cache Hit Rate % stat card linking to /tokens
- **New**: Top Conversation Cost stat card linking to /conversations

### Changed
- Dashboard now serves as entry point to all insight pages rather than standalone data dump
- All 4 dashboard stat cards are now clickable links
- Default dashboard period changed from 1hour to today
- Replaced inline filter buttons with select dropdown and custom date inputs
- Legacy 1hour preset mapped to 15min (closest semantic match)
- Improved navigation hub grid for Tools & Features, Sub-Agents, Files, Integrations, Conversations, Projects

### Fixed
- **Critical**: Fixed UTC-aware datetime handling throughout time filtering to prevent naive/aware comparison crashes
- Fixed end_dt to 23:59:59.999999 to include fractional seconds in day ranges
- Fixed day label generation to use UTC date objects, eliminating timezone mismatches
- Fixed dashboard to use UTC-aware datetime.min fallback in sort operations
- Fixed usage.py to correctly convert local naive datetimes to UTC-aware using fromtimestamp()
- Guard custom range requests against incomplete date inputs
- Normalize legacy stored periods (week, month, 1hour) to new presets on restore
- Guard peak_context_tokens display against None values
- Improved accessibility with aria-labels on time filter controls
- Replaced onclick table rows with accessible anchor links

### Removed
- Removed redundant Usage Overview and Token Summary tables from dashboard
- Removed 1hour preset (mapped to 15min for backward compatibility)

## [2.6.0] - 2026-02-28

### Added
- **New**: Progress indicators on time filter buttons across all pages
- **New**: Inline spinner icons for better visual feedback during data loading
- **Enhancement**: Improved UX with immediate feedback when switching time ranges (15m, 1h, Today, Week, Month, All)

### Changed
- Updated all 7 metric pages with loading indicators (tokens, projects, files, integrations, features, subagents, teams)
- Enhanced button layout with proper spinner positioning using flexbox
- Improved HTMX integration with `hx-indicator` configuration for seamless loading states

### Technical Details
- Added `htmx-indicator` class to indicator `<span>` elements inside time filter buttons
- Configured buttons with `hx-indicator="this"` for inline spinner display
- Updated programmatic HTMX calls to pass clicked button as indicator target
- Leveraged existing CSS animations for spinner rotation and visibility

## [2.1.0] - 2025-12-31

### Added
- **New**: Customizable model pricing configuration
- **New**: Settings page at `/settings` to configure pricing per model
- **New**: API endpoints for getting and updating model pricing (`GET/POST /api/pricing`, `POST /api/pricing/reset`)
- **New**: Dynamic model discovery - settings page only shows models that have been used (no hardcoded models)
- **New**: Pricing data stored in `pricing.json` for persistence
- **New**: Default pricing ($3.00 input, $15.00 output per million tokens) for unconfigured models

### Changed
- Updated token calculation to use custom pricing when available
- Updated dashboard service to use custom pricing throughout
- Added settings link to navigation
- Removed hardcoded model list - now truly flexible for any LLM models

### Technical Details
- Added `pricing.py` utility module for pricing management
- Models are dynamically discovered from session data
- Any new models automatically appear with default pricing
- Works with any models (Claude, GLM, local LLMs, etc.)

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
