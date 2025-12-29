<div align="center">

# ClaudeSavvy

### The missing dashboard for Claude Code — track costs, optimize usage, stay in control.

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Maintained](https://img.shields.io/badge/Maintained%3F-yes-green.svg)

<img src="docs/images/hero-viewport.png" alt="ClaudeSavvy Dashboard" width="800">

</div>

---

## Overview

ClaudeSavvy provides comprehensive analytics for your Claude Code usage by reading local data files and presenting insights through a clean web interface. No API keys required, no data sent externally—everything runs locally on your machine.

**Key Benefits:**
- 📊 **Visual Analytics**: Modern web dashboard with detailed metrics
- 💰 **Cost Tracking**: Monitor spending and cache savings in real-time
- 🚀 **Performance Insights**: Identify optimization opportunities
- 📁 **Project Breakdown**: See which projects consume the most resources
- 🔧 **Tool Analytics**: Track sub-agent and MCP integration usage

## Features

### Dashboard Pages

- **Overview Dashboard**: At-a-glance summary of all key metrics
- **Token Usage**: Detailed breakdown of input, output, cache reads/writes with cost calculations
- **Projects**: Per-project analytics showing sessions, commands, and token usage
- **Files**: Most frequently edited files and operation statistics
- **Integrations**: MCP server usage and activity tracking
- **Features**: Tool usage, sub-agents, skills, and configuration analytics

### Capabilities

- **Real-time Monitoring**: Live data from your Claude Code usage
- **Cost Analysis**: Track spending with Claude Sonnet 4.5 pricing
- **Cache Efficiency**: Monitor prompt cache performance and savings
- **Time Filtering**: View metrics by today, week, month, or all-time
- **Data Export**: Export metrics to CSV or JSON format
- **Zero Configuration**: Works automatically with Claude Code data

## Quick Start

### Installation

**Install via PyPI (Recommended):**

```bash
pip install claudesavvy
```

**Install from source (for development):**

```bash
# Clone the repository
git clone https://github.com/allannapier/claude_monitor.git
cd claude_monitor

# Install the package
pip install .
```

### Running

```bash
# Start the web server (default port 5000)
claudesavvy

# Open your browser to http://localhost:5000
```

That's it! The dashboard will load your Claude Code usage data automatically.

## Usage

### Basic Commands

```bash
# Start server on default port 5000
claudesavvy

# Start on custom port
claudesavvy --port 8080

# Bind to all network interfaces (allow external access)
claudesavvy --host 0.0.0.0

# Enable debug mode with auto-reload
claudesavvy --debug

# Use custom Claude data directory
claudesavvy --claude-dir /path/to/claude/data
```

### Web Interface

Once the server is running, navigate to:

- **http://localhost:5000** - Main dashboard
- **http://localhost:5000/tokens** - Token usage details
- **http://localhost:5000/projects** - Project analytics
- **http://localhost:5000/files** - File operation stats
- **http://localhost:5000/integrations** - MCP integrations
- **http://localhost:5000/features** - Tool and feature usage

### Data Export

Export your metrics for further analysis:

- **CSV Export**: http://localhost:5000/export/csv
- **JSON Export**: http://localhost:5000/export/json

## Screenshots

### Dashboard Overview
The main dashboard provides an at-a-glance view of all your Claude Code metrics.

![Dashboard](docs/images/dashboard.png)

### Token Usage & Costs
Detailed breakdown of token consumption with cost calculations and cache efficiency.

![Token Usage](docs/images/tokens.png)

### Features & Tools
Track tool usage, sub-agents, MCP integrations, and installed skills.

![Features](docs/images/features.png)

## Metrics Explained

### Usage Metrics
- **Total Sessions**: Number of unique Claude Code sessions
- **Total Commands**: Number of commands/queries sent to Claude
- **Active Projects**: Number of different projects worked on
- **Time Range**: Filtered time period for displayed metrics

### Token Metrics
- **Input Tokens**: Regular input tokens (non-cached)
- **Output Tokens**: Tokens in Claude's responses
- **Cache Write**: Tokens written to prompt cache
- **Cache Read**: Tokens read from prompt cache (90% cheaper!)
- **Cache Efficiency**: Percentage of cache reads vs. cache writes

### Cost Calculations

Based on Claude Sonnet 4.5 pricing:
- **Input**: $3.00 per million tokens
- **Output**: $15.00 per million tokens
- **Cache Write**: $3.75 per million tokens
- **Cache Read**: $0.30 per million tokens (90% discount!)

### Cache Savings

Shows how much money you've saved through prompt caching. The calculation compares what cache reads would have cost as regular input tokens versus the actual discounted cache read cost. High cache efficiency (>90%) means significant savings!

## Data Sources

ClaudeSavvy reads data from your local Claude Code installation at `~/.claude/`:

- **history.jsonl**: Command history and timestamps
- **projects/**: Session data with detailed token usage
- **debug/**: MCP server activity logs
- **file-history/**: File modification tracking
- **skills/**: Installed skills and configurations
- **settings.json**: Claude Code settings

**Privacy**: All data stays on your machine. No external API calls, no data uploaded.

## Requirements

- **Python**: 3.9 or higher
- **Claude Code**: Must be installed and used at least once
- **Dependencies**:
  - click>=8.1.0
  - rich>=13.0.0
  - python-dateutil>=2.8.0
  - Flask>=3.0.0
  - Jinja2>=3.1.0

## Project Structure

```
claude_monitor/
├── src/claudesavvy/
│   ├── cli.py              # Entry point - launches web server
│   ├── parsers/            # Data parsers for Claude files
│   │   ├── history.py      # Command history parser
│   │   ├── sessions.py     # Session and token data parser
│   │   ├── debug.py        # MCP server logs parser
│   │   ├── files.py        # File editing history parser
│   │   ├── tools.py        # Tool usage parser
│   │   └── skills.py       # Skills and config parser
│   ├── analyzers/          # Data analysis and aggregation
│   │   ├── usage.py        # Usage statistics analyzer
│   │   ├── tokens.py       # Token usage and cost analyzer
│   │   ├── integrations.py # MCP integration analyzer
│   │   └── features.py     # Features analyzer
│   ├── web/                # Flask web application
│   │   ├── app.py          # Flask app factory
│   │   ├── routes/         # Route handlers
│   │   ├── services/       # Business logic layer
│   │   ├── templates/      # Jinja2 HTML templates
│   │   └── static/         # CSS, JS, images
│   └── utils/              # Shared utilities
│       ├── paths.py        # Path management
│       └── time_filter.py  # Time-based filtering
├── LICENSE                 # MIT License
├── README.md              # This file
├── CONTRIBUTING.md        # Contribution guidelines
├── CHANGELOG.md           # Version history
├── setup.py               # Package setup
├── pyproject.toml         # Modern Python packaging
└── requirements.txt       # Dependencies
```

## Development

Want to contribute? Check out [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Run in Development Mode

```bash
# Install in editable mode
pip install -e .

# Run with debug mode (auto-reload on code changes)
claudesavvy --debug
```

### Run Tests

```bash
# Manual testing - verify all routes work
claudesavvy
# Then navigate to each page in your browser
```

## Tips & Best Practices

1. **Monitor Cache Efficiency**: A high cache efficiency (>90%) means you're saving money through prompt caching. This happens when working on the same project/context repeatedly.

2. **Track Costs**: Use the Tokens page to monitor your spending. The dashboard shows both gross cost and net cost after cache savings.

3. **Analyze Projects**: The Projects page helps identify which projects consume the most tokens and cost the most, useful for budgeting.

4. **Time Filtering**: Use the time period selector to track recent activity and costs, or view all-time statistics.

5. **Export Data**: Use CSV/JSON export for custom analysis, charting in Excel, or integration with other tools.

6. **Performance**: The tool uses streaming parsing to handle large session files efficiently, even with extensive Claude Code usage.

## Troubleshooting

### "FileNotFoundError: ~/.claude/ not found"
- Ensure Claude Code is installed
- Run Claude Code at least once to generate data files

### "ImportError: No module named 'flask'"
- Install dependencies: `pip install -r requirements.txt`
- Or reinstall the package: `pip install .`

### Web server won't start
- Check if port 5000 is already in use
- Try a different port: `claudesavvy --port 8080`
- Check Python version: `python3 --version` (requires 3.9+)

### Data not showing
- Verify Claude Code has been used recently
- Check `~/.claude/` directory exists and contains data files
- Try restarting the web server

## Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

- **v1.0.0** (2025-12-20): Web-only release with production polish
- **v0.1.0** (2025-12-19): Initial release with CLI dashboard

## License

MIT License - see [LICENSE](LICENSE) file for details.

**THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.** See LICENSE for full terms.

## Author

**Allan Napier**

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

- **Issues**: [GitHub Issues](https://github.com/allannapier/claude_monitor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/allannapier/claude_monitor/discussions)

---

**Note**: This tool is not affiliated with or endorsed by Anthropic. It's an independent monitoring tool for Claude Code users.
