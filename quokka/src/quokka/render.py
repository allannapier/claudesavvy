"""Render the harness graph as a self-contained HTML file."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_html(graph: dict[str, Any]) -> str:
    """Return a complete HTML document that visualises ``graph``."""
    env = _env()
    template = env.get_template("viewer.html.j2")
    return template.render(
        graph=graph,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
