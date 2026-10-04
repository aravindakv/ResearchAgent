"""Markdown -> sanitized HTML -> PDF. Temporary: moves to the MCP server in chapter 05."""
import markdown as md_lib
import nh3
from weasyprint import HTML

ALLOWED_TAGS = {"h1", "h2", "h3", "h4", "p", "ul", "ol", "li", "strong", "em", "code", "pre", "blockquote",
                "table", "thead", "tbody", "tr", "th", "td", "a", "br", "hr"}
PDF_CSS = """
@page { size: A4; margin: 2cm; @bottom-center { content: counter(page); font-size: 9pt; } }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 10.5pt; line-height: 1.5; }
h1 { font-size: 20pt; } h2 { font-size: 14pt; margin-top: 1.4em; }
pre, code { font-family: 'DejaVu Sans Mono', monospace; font-size: 9pt; }
pre { background: #f4f4f4; padding: 8px; white-space: pre-wrap; }
blockquote { border-left: 3px solid #c77; margin: 0; padding-left: 10px; color: #733; }
table { border-collapse: collapse; } td, th { border: 1px solid #ccc; padding: 4px; }
"""

def deny_fetch(url, *args, **kwargs):
    raise ValueError(f"fetching URL is denied: {url}")

def markdown_to_pdf(markdown_text: str) -> bytes:
    body = nh3.clean(md_lib.markdown(markdown_text, extensions=["tables", "fenced_code"]), tags=ALLOWED_TAGS)
    document = f"<html><head><meta charset='utf-8'><style>{PDF_CSS}</style></head><body>{body}</body></html>"
    return HTML(string=document, url_fetcher=deny_fetch).write_pdf()
