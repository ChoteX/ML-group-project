from __future__ import annotations

import argparse
import html
from pathlib import Path

import markdown
from pygments.formatters import HtmlFormatter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "docs" / "report.md"
DEFAULT_OUTPUT = ROOT / "docs" / "report.html"


BASE_CSS = """
:root {
  color-scheme: light;
  --bg: #ffffff;
  --fg: #1f2328;
  --muted: #59636e;
  --border: #d0d7de;
  --border-muted: #d8dee4;
  --canvas-subtle: #f6f8fa;
  --accent: #0969da;
  --blockquote: #656d76;
  --shadow: 0 1px 2px rgba(31, 35, 40, 0.04);
}

* {
  box-sizing: border-box;
}

html,
body {
  margin: 0;
  padding: 0;
  background: var(--canvas-subtle);
  color: var(--fg);
}

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  line-height: 1.6;
  text-rendering: optimizeLegibility;
}

a {
  color: var(--accent);
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}

.page {
  padding: 32px 20px 56px;
}

.markdown-body {
  max-width: 980px;
  margin: 0 auto;
  padding: 40px 48px;
  background: var(--bg);
  border: 1px solid var(--border-muted);
  border-radius: 12px;
  box-shadow: var(--shadow);
  word-wrap: break-word;
}

.markdown-body > *:first-child {
  margin-top: 0 !important;
}

.markdown-body > *:last-child {
  margin-bottom: 0 !important;
}

.markdown-body h1,
.markdown-body h2,
.markdown-body h3,
.markdown-body h4,
.markdown-body h5,
.markdown-body h6 {
  margin-top: 24px;
  margin-bottom: 16px;
  font-weight: 600;
  line-height: 1.25;
}

.markdown-body h1 {
  margin-top: 0;
  padding-bottom: 0.3em;
  font-size: 2em;
  border-bottom: 1px solid var(--border);
}

.markdown-body h2 {
  padding-bottom: 0.3em;
  font-size: 1.5em;
  border-bottom: 1px solid var(--border);
}

.markdown-body h3 {
  font-size: 1.25em;
}

.markdown-body p,
.markdown-body ul,
.markdown-body ol,
.markdown-body table,
.markdown-body blockquote,
.markdown-body pre {
  margin-top: 0;
  margin-bottom: 16px;
}

.markdown-body ul,
.markdown-body ol {
  padding-left: 2em;
}

.markdown-body li + li {
  margin-top: 0.25em;
}

.markdown-body li > p {
  margin-top: 16px;
}

.markdown-body hr {
  height: 0.25em;
  margin: 24px 0;
  padding: 0;
  background: var(--border);
  border: 0;
}

.markdown-body blockquote {
  margin-left: 0;
  padding: 0 1em;
  color: var(--blockquote);
  border-left: 0.25em solid var(--border);
}

.markdown-body table {
  display: block;
  width: max-content;
  max-width: 100%;
  overflow: auto;
  border-spacing: 0;
  border-collapse: collapse;
}

.markdown-body table th,
.markdown-body table td {
  padding: 6px 13px;
  border: 1px solid var(--border);
}

.markdown-body table th {
  font-weight: 600;
  background: var(--canvas-subtle);
}

.markdown-body table tr:nth-child(2n) {
  background: #f6f8fa;
}

.markdown-body code,
.markdown-body tt {
  padding: 0.2em 0.4em;
  margin: 0;
  font-family: ui-monospace, SFMono-Regular, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-size: 85%;
  background: rgba(175, 184, 193, 0.2);
  border-radius: 6px;
}

.markdown-body pre {
  padding: 16px;
  overflow: auto;
  font-size: 85%;
  line-height: 1.45;
  background: #f6f8fa;
  border-radius: 6px;
}

.markdown-body pre code {
  padding: 0;
  background: transparent;
  border-radius: 0;
}

.markdown-body img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 20px auto 10px;
}

.markdown-body strong {
  font-weight: 600;
}

.markdown-body em {
  color: var(--muted);
}

.markdown-body .footnote {
  color: var(--muted);
  font-size: 0.95em;
}

@media (max-width: 840px) {
  .page {
    padding: 16px 10px 24px;
  }

  .markdown-body {
    padding: 24px 18px;
    border-radius: 8px;
  }
}

@media print {
  body,
  html {
    background: #fff;
  }

  .page {
    padding: 0;
  }

  .markdown-body {
    max-width: none;
    border: 0;
    box-shadow: none;
    padding: 0;
  }
}
"""


def build_html_document(title: str, body_html: str, code_css: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{html.escape(title)}</title>
    <style>
{BASE_CSS}
{code_css}
    </style>
  </head>
  <body>
    <div class="page">
      <article class="markdown-body">
{body_html}
      </article>
    </div>
  </body>
</html>
"""


def render_markdown(markdown_text: str) -> str:
    renderer = markdown.Markdown(
        extensions=[
            "extra",
            "tables",
            "fenced_code",
            "codehilite",
            "sane_lists",
            "toc",
        ],
        extension_configs={
            "codehilite": {
                "guess_lang": False,
                "linenums": False,
                "css_class": "codehilite",
            },
            "toc": {
                "permalink": False,
            },
        },
        output_format="html5",
    )
    return renderer.convert(markdown_text)


def extract_title(markdown_text: str, input_path: Path) -> str:
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return input_path.stem


def convert_markdown_to_html(input_path: Path, output_path: Path) -> None:
    markdown_text = input_path.read_text(encoding="utf-8")
    title = extract_title(markdown_text, input_path)
    body_html = render_markdown(markdown_text)
    code_css = HtmlFormatter(style="default").get_style_defs(".codehilite")
    html_document = build_html_document(title, body_html, code_css)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Markdown report to a standalone HTML file with GitHub-like styling."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Markdown input file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"HTML output file. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    convert_markdown_to_html(input_path, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
