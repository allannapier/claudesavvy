# Quokka — Claude Code Harness Viewer

Quokka scans every file that influences a Claude Code session — skills, MCP
servers, sub-agents, slash commands, hooks, plugins, `CLAUDE.md` /
`AGENTS.md` context docs, and `settings.json` files at both **user** and
**project** scope — and renders an interactive node-link graph of how they
interlink.

The output is a single self-contained HTML file. No server, no API keys, no
data leaves your machine.

## Install

```bash
pip install -e .
```

## Run

```bash
# Scan the current directory + ~/.claude/, write ./harness.html
quokka

# Scan a specific project, custom output, open in browser
quokka --repo /path/to/project --out /tmp/harness.html --open
```

Or without installing:

```bash
python -m quokka --repo .
```

## What you get

* **Nodes** for every Skill, Agent, Command, Hook, MCP server, Plugin,
  context-doc (CLAUDE.md / AGENTS.md), and settings file found.
* **Edges** for the relationships between them:
  * `provides` — Plugin → its Skill / Agent / Command / Hook / MCP
  * `triggers_on` — Hook → the tool (or event) that fires it
  * `grants_tool` — Agent → the tools / MCPs listed in its frontmatter
  * `references` — context-doc → entities mentioned by name or `/slash`
* Click any node to see its source path, frontmatter, and content.
* Filter chips toggle visibility per type and per source.

## Layout

```
quokka/
├── pyproject.toml
├── src/quokka/
│   ├── scanner.py     # filesystem discovery
│   ├── references.py  # text mining for context-doc references
│   ├── graph.py       # node + edge construction
│   ├── render.py      # Jinja2 → standalone HTML
│   ├── cli.py         # argparse entry point
│   └── templates/
│       └── viewer.html.j2
└── tests/
```
