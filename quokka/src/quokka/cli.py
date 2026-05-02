"""Command-line entry point for Quokka."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from collections.abc import Sequence
from pathlib import Path

from .graph import build_graph
from .render import render_html
from .scanner import ConfigScanner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quokka",
        description=(
            "Scan a project's Claude Code harness (skills, MCPs, agents, hooks, "
            "slash commands, plugins, CLAUDE.md / AGENTS.md, settings) and emit "
            "a self-contained HTML graph viewer."
        ),
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Project directory to inspect (default: current working directory).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("harness.html"),
        help="Output HTML file (default: ./harness.html).",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the resulting HTML file in the default browser.",
    )
    parser.add_argument(
        "--user-home",
        type=Path,
        default=None,
        help="Override the user home directory (mainly for testing).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo = args.repo.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    if not repo.exists():
        print(f"error: repo path does not exist: {repo}", file=sys.stderr)
        return 2

    scanner = ConfigScanner(user_home=args.user_home)
    data = scanner.scan(repo)
    graph = build_graph(data)
    html = render_html(graph)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    counts = data.counts()
    summary = ", ".join(f"{v} {k}" for k, v in counts.items() if v)
    if not summary:
        summary = "no harness artifacts found"
    print(f"Wrote {out_path}  ({summary})")

    if args.open:
        webbrowser.open(out_path.as_uri())

    return 0
