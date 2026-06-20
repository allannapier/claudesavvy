#!/usr/bin/env python3
"""Render ClaudeSavvy demo pages to static HTML for GitHub Pages.

Usage:
    python scripts/render_static.py [--output docs/demo]

Produces a self-contained directory of HTML files with:
- All main dashboard pages rendered with fake demo data
- Static assets (JS/CSS) copied alongside
- Navigation links rewritten to work as plain .html files
- A "static demo" notice injected into each page

The output directory can be committed and served via GitHub Pages.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

# Make sure the src package is importable when run from the project root
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))


ROUTES = [
    ("/",             "index.html",        "Dashboard"),
    ("/tokens",       "tokens.html",       "Tokens"),
    ("/projects",     "projects.html",     "Projects"),
    ("/files",        "files.html",        "Files"),
    ("/integrations", "integrations.html", "Integrations"),
    ("/features",     "features.html",     "Features"),
    ("/conversations","conversations.html","Conversations"),
]

# Nav links that appear in every page sidebar
_NAV_REWRITES = {
    'href="/"':              'href="index.html"',
    'href="/dashboard"':     'href="index.html"',
    'href="/tokens"':        'href="tokens.html"',
    'href="/projects"':      'href="projects.html"',
    'href="/files"':         'href="files.html"',
    'href="/integrations"':  'href="integrations.html"',
    'href="/features"':      'href="features.html"',
    'href="/conversations"': 'href="conversations.html"',
    # These pages exist but aren't in the static export
    'href="/harness"':       'href="#"',
    'href="/subagents"':     'href="#"',
    'href="/teams"':         'href="#"',
    'href="/configuration"': 'href="#"',
    'href="/settings"':      'href="#"',
}

_DEMO_BANNER = """\
<div id="demo-banner" style="
    position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
    background: #0770E3; color: #fff;
    padding: 8px 16px; font-size: 13px;
    display: flex; align-items: center; justify-content: center; gap: 12px;
    font-family: system-ui, sans-serif;
">
    <span>&#9432; Static demo &mdash; showing sample data.</span>
    <a href="https://github.com/allannapier/claudesavvy#quick-start"
       style="color:#fff; font-weight:600; text-decoration:underline;"
       target="_blank">Install ClaudeSavvy</a>
    <span>for live data and interactive filtering.</span>
</div>
<style>
    /* Push page content below the fixed banner */
    body { padding-top: 38px !important; }
    /* Hide the HTMX time-filter controls in static mode */
    [hx-get], [data-hx-get] { pointer-events: none; opacity: 0.5; }
</style>
"""

_DISABLE_HTMX = """\
<script>
/* Prevent HTMX API calls in static demo */
document.addEventListener('htmx:beforeRequest', function(e) { e.preventDefault(); });
</script>
"""


def _rewrite_html(html: str) -> str:
    """Fix up URLs and inject demo chrome into a rendered page."""
    # Static assets: make relative so they work from any hosting path
    html = html.replace('src="/static/', 'src="static/')
    html = html.replace('href="/static/', 'href="static/')

    # Navigation links
    for old, new in _NAV_REWRITES.items():
        html = html.replace(old, new)

    # Export links won't work in static mode
    html = html.replace('href="/export/csv"', 'href="#"')
    html = html.replace('href="/export/json"', 'href="#"')

    # Inject demo banner + HTMX disable just after <body>
    html = re.sub(r'(<body[^>]*>)', r'\1' + _DEMO_BANNER, html, count=1)

    # Inject HTMX disable script just before </body>
    html = html.replace('</body>', _DISABLE_HTMX + '</body>', 1)

    return html


def render(output_dir: Path) -> None:
    from claudesavvy.data.demo import generate_demo_data
    from claudesavvy.web.app import create_app

    print("Generating demo data...")
    demo_paths = generate_demo_data()

    print("Creating Flask app...")
    app = create_app(demo_paths)

    output_dir.mkdir(parents=True, exist_ok=True)

    with app.test_client() as client:
        for route, filename, label in ROUTES:
            resp = client.get(f"{route}?period=all")
            if resp.status_code == 200:
                html = _rewrite_html(resp.data.decode("utf-8", errors="replace"))
                (output_dir / filename).write_text(html, encoding="utf-8")
                print(f"  OK  {route:25s} -> {filename}")
            else:
                print(f"  ERR {route:25s} -> {resp.status_code}")

    # Copy static assets
    static_src = _ROOT / "src" / "claudesavvy" / "web" / "static"
    static_dst = output_dir / "static"
    if static_src.exists():
        if static_dst.exists():
            shutil.rmtree(static_dst)
        shutil.copytree(static_src, static_dst)
        print(f"  Copied static assets -> {static_dst}")

    print(f"\nDone. Output: {output_dir}/")
    print("Commit the docs/ directory and enable GitHub Pages (source: docs/).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output",
        default="docs/demo",
        help="Output directory (default: docs/demo)",
    )
    args = parser.parse_args()
    render(Path(args.output))


if __name__ == "__main__":
    main()
