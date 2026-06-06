"""Render THESIS_REPORT_FINAL.md into a single self-contained HTML file with all
images embedded as base64, styled like an academic report. Open the result in a
browser and use Print -> Save as PDF to obtain the final document.

No external converter (pandoc/weasyprint/wkhtmltopdf) is required — only the
pure-Python `markdown` package.
"""
from __future__ import annotations
import base64
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).parent.parent
MD = ROOT / "THESIS_REPORT_FINAL.md"
OUT = ROOT / "THESIS_REPORT_FINAL.html"


def embed_images(md_text: str) -> str:
    """Replace ![alt](path) image paths with base64 data URIs."""
    def repl(m):
        alt, path = m.group(1), m.group(2)
        p = (ROOT / path).resolve()
        if not p.exists():
            return m.group(0)
        data = base64.b64encode(p.read_bytes()).decode("ascii")
        ext = p.suffix.lstrip(".").lower()
        mime = "image/png" if ext == "png" else f"image/{ext}"
        return f'![{alt}](data:{mime};base64,{data})'
    return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', repl, md_text)


CSS = """
@page { size: A4; margin: 22mm 20mm; }
body {
  font-family: "Georgia", "Times New Roman", serif;
  font-size: 11.5pt; line-height: 1.55; color: #1a1a1a;
  max-width: 820px; margin: 0 auto; padding: 24px;
}
h1 { font-size: 20pt; margin-top: 1.6em; border-bottom: 2px solid #333; padding-bottom: 6px; }
h2 { font-size: 15pt; margin-top: 1.4em; color: #1f2937; }
h3 { font-size: 12.5pt; margin-top: 1.1em; color: #374151; }
p { text-align: justify; }
img { max-width: 100%; display: block; margin: 14px auto; border: 1px solid #e5e7eb; }
table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 10pt; }
th, td { border: 1px solid #cbd5e1; padding: 6px 10px; text-align: left; }
th { background: #f1f5f9; font-weight: 600; }
td[style*="border:none"], td[style*="border: none"] { border: none !important; }
table[style*="border:none"], table[style*="border: none"] { border: none !important; }
blockquote {
  border-left: 3px solid #0d9488; margin: 12px 0; padding: 6px 16px;
  background: #f8fafc; font-family: "Courier New", monospace; font-size: 10.5pt;
}
code { font-family: "Courier New", monospace; background: #f1f5f9; padding: 1px 4px; }
div[style*="page-break-after"] { page-break-after: always; }
strong { color: #111827; }
a { color: #0d9488; text-decoration: none; }
"""


def main():
    text = MD.read_text(encoding="utf-8")
    text = embed_images(text)
    html_body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "attr_list", "sane_lists"]
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Surgical Phase Recognition — Bachelor's Thesis</title>
<style>{CSS}</style></head>
<body>{html_body}</body></html>"""
    OUT.write_text(html, encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"Wrote {OUT} ({size_kb:.0f} KB)")
    print("Open in a browser and Print -> Save as PDF for the final document.")


if __name__ == "__main__":
    main()
