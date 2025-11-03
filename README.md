# Claude Monitor

A comprehensive CLI tool for tracking and analyzing your Claude Code usage metrics.

## Features

- **Interactive Menu**: Beautiful entry page with logo and menu for easy navigation
- **Usage Metrics**: Track sessions, commands, messages, and active projects
- **Token Analysis**: Detailed token usage breakdown with cost calculations
- **Cache Efficiency**: Monitor prompt caching performance and savings
- **Project Breakdown**: Per-project metrics and activity tracking
- **MCP Integrations**: Track usage of MCP servers (GitHub, Atlassian, etc.)
- **File Editing**: See your most frequently edited files
- **Time Filtering**: Filter by today, week, month, year, or custom date ranges
- **Rich CLI Display**: Beautiful tables, charts, and color-coded output

## Installation

```bash
# Clone the repository
git clone https://github.com/Skyscanner/claude_monitor.git
cd claude_monitor

# Install dependencies
pip install -r requirements.txt
```

Alternatively, you can install it as a package:

```bash
# Install in editable mode (recommended for development)
pip install -e .

# Or install normally
pip install .
```

After installation, you can run it directly with:
```bash
claude-monitor
```

## Usage

### Interactive Mode (Recommended)

Simply run the tool without arguments to launch the interactive menu:

```bash
python3 -m src.claude_monitor.cli
```

The interactive menu features:
- **Claude Monitor Logo**: Styled in Skyscanner blue
- **View Selection**: Choose from Full Dashboard, Token Usage, Activity Report, Integrations, or File History
- **Time Range Selection**: Today, Last 7 days, Last 30 days, Current quarter, Last year, or All time

### Command Line Options

You can also use CLI flags for direct access:

```bash
# Show full dashboard (all-time)
python3 -m src.claude_monitor.cli --no-interactive

# Show last 7 days
python3 -m src.claude_monitor.cli --week

# Show last 30 days
python3 -m src.claude_monitor.cli --month

# Show current quarter (Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec)
python3 -m src.claude_monitor.cli --quarter

# Show only today
python3 -m src.claude_monitor.cli --today

# Show since a specific date
python3 -m src.claude_monitor.cli --since 2025-01-01
```

### Focus on Specific Metrics

```bash
# Show only token usage and costs
python3 -m src.claude_monitor.cli --focus tokens

# Show only activity metrics
python3 -m src.claude_monitor.cli --focus usage

# Show only MCP integrations
python3 -m src.claude_monitor.cli --focus integrations

# Show only file editing stats
python3 -m src.claude_monitor.cli --focus files
```

### Filter by Project

```bash
# Filter to specific project
python3 -m src.claude_monitor.cli --project /path/to/project
```

## Sample Output

The dashboard features a modern, space-efficient layout with:
- **Summary Cards**: Key metrics at a glance (sessions, tokens, costs, cache efficiency)
- **Side-by-Side Tables**: Detailed usage and token metrics using full terminal width
- **Project Breakdown**: See your most active projects
- **File Editing Stats**: Track your most frequently edited files

```
╭──────────────────────────────────────────────────────────────────────────────╮
│ Claude Code Usage Dashboard                                                  │
│ (Last 30 days)                                                               │
╰──────────────────────────────────────────────────────────────────────────────╯

╭───────────────────╮  ╭───────────────────╮  ╭───────────────────╮  ╭──────────────────╮
│  SESSIONS         │  │   TOTAL TOKENS    │  │  TOTAL COST       │  │  CACHE HIT RATE  │
│  259              │  │   1.7B            │  │  $975.85          │  │  92.9%           │
│  1,481 commands   │  │                   │  │  -$4335.89 saved  │  │  Excellent       │
╰───────────────────╯  ╰───────────────────╯  ╰───────────────────╯  ╰──────────────────╯

╭─────────────────────────────────────╮  ╭─────────────────────────────────────╮
│  📊 USAGE SUMMARY                   │  │  💰 TOKEN USAGE & COSTS             │
│  ┏━━━━━━━━━━━━━┳━━━━━━━━┓            │  │  ┏━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓  │
│  ┃ Metric      ┃  Value ┃            │  │  ┃ Token Type ┃  Count ┃   Cost ┃  │
│  ┡━━━━━━━━━━━━━╇━━━━━━━━┩            │  │  ┡━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩  │
│  │ Sessions    │    259 │            │  │  │ Input      │  905K  │  $2.71 │  │
│  │ Commands    │  1,481 │            │  │  │ Output     │  1.9M  │ $28.29 │  │
│  │ Projects    │     19 │            │  │  │ Cache Write│ 123.5M │$463.08 │  │
│  └─────────────┴────────┘            │  │  │ Cache Read │1605.9M │$481.77 │  │
╰─────────────────────────────────────╯  │  │ TOTAL      │1732.2M │$975.85 │  │
                                         │  │ Cache Savings      │-$4335.89│  │
                                         │  └────────────────────────────────┘  │
                                         │  Cache Efficiency: ██████████ 92.9% │
                                         ╰─────────────────────────────────────╯

╭─────────────────────────────────────────────────────────────────────────────╮
│  📁 TOP PROJECTS                                                            │
│  ┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓                           │
│  ┃ Project               ┃ Sessions ┃ Commands ┃                           │
│  ┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩                           │
│  │ ads_gf_bot            │       50 │      861 │                           │
│  │ databricks-pelai      │       82 │      320 │                           │
│  │ askclaude_app         │        7 │       94 │                           │
│  └───────────────────────┴──────────┴──────────┘                           │
╰─────────────────────────────────────────────────────────────────────────────╯
```

## Metrics Explained

### Usage Metrics
- **Total Sessions**: Number of unique Claude Code sessions
- **Total Commands**: Number of commands/queries you've sent
- **Total Messages**: Total back-and-forth messages (including Claude's responses)
- **Active Projects**: Number of different projects you've worked on
- **Avg Commands/Day**: Average daily command count

### Token Metrics
- **Input Tokens**: Regular input tokens (non-cached)
- **Output Tokens**: Tokens in Claude's responses
- **Cache Write**: Tokens written to prompt cache
- **Cache Read**: Tokens read from prompt cache (much cheaper!)
- **Cache Efficiency**: Percentage of cache reads vs. cache writes

### Cost Calculations
Based on Claude Sonnet 4.5 pricing:
- Input: $3/million tokens
- Output: $15/million tokens
- Cache Write: $3.75/million tokens
- Cache Read: $0.30/million tokens (90% discount!)

### Cache Savings
Shows how much money you saved by using prompt caching. The calculation compares what cache reads would have cost as regular input tokens vs. the actual discounted cache read cost.

## Data Sources

The tool analyzes local Claude Code data from `~/.claude/`:

- **history.jsonl**: Command history and timestamps
- **projects/**: Session data with token usage
- **debug/**: MCP server activity logs
- **file-history/**: File modification tracking

No API keys or external services required!

## Requirements

- Python 3.9+
- Claude Code CLI installed and used at least once
- Dependencies: click, rich, python-dateutil

## Project Structure

```
claude_monitor/
├── src/claude_monitor/
│   ├── cli.py              # Main CLI entry point
│   ├── parsers/            # Data parsers for Claude files
│   │   ├── history.py      # Command history parser
│   │   ├── sessions.py     # Session and token data parser
│   │   ├── debug.py        # MCP server logs parser
│   │   └── files.py        # File editing history parser
│   ├── analyzers/          # Data analysis and aggregation
│   │   ├── usage.py        # Usage statistics analyzer
│   │   ├── tokens.py       # Token usage and cost analyzer
│   │   └── integrations.py # MCP integration analyzer
│   ├── display/            # Rich CLI formatting
│   │   ├── formatter.py    # Display formatting utilities
│   │   ├── tables.py       # Table builders
│   │   └── dashboard.py    # Main dashboard
│   └── utils/              # Shared utilities
│       ├── paths.py        # Path management
│       └── time_filter.py  # Time-based filtering
├── pyproject.toml
└── README.md
```

## Tips

1. **Cache Efficiency**: A high cache efficiency (>90%) means you're saving significant money through prompt caching. This happens when you're working on the same project/context repeatedly.

2. **Cost Tracking**: Use `--focus tokens` to monitor your spending. The tool shows both gross cost and net cost after cache savings.

3. **Project Analysis**: Use the project breakdown to see which projects consume the most tokens and cost the most.

4. **Time Filtering**: Use `--week` or `--month` to track recent activity and costs, or `--since` for custom date ranges.

5. **Performance**: The tool uses streaming parsing to handle large session files efficiently, so it should work well even with extensive Claude Code usage.

## License

MIT License - Feel free to use and modify as needed!

## Author

Allan Napier
