#!/usr/bin/env python
"""
Render paper/manuscript.md to a standalone PDF (paper/manuscript.pdf).

Pure-Python (markdown + xhtml2pdf/reportlab) so it works without pandoc/LaTeX.
- Converts LaTeX \citep/\citet commands to readable (Author Year) citations using
  references.bib.
- Embeds DejaVuSans (from matplotlib) for full Unicode coverage (phi, sigma, M_sun...).
- Resolves figure paths relative to paper/.
"""
from __future__ import annotations
import os
import re
from pathlib import Path

import matplotlib
import markdown
from xhtml2pdf import pisa
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

HERE = Path(__file__).resolve().parent
MD = HERE / "manuscript.md"
BIB = HERE / "references.bib"
OUT = HERE / "manuscript.pdf"
FONTDIR = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"


def parse_bib(path: Path) -> dict[str, str]:
    """key -> 'Surname Year' (or 'Surname et al. Year')."""
    text = path.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", text, re.S):
        key = m.group(1).strip()
        body = m.group(2)
        am = re.search(r"author\s*=\s*\{(.+?)\}", body, re.S)
        ym = re.search(r"year\s*=\s*\{?(\d{4})", body)
        year = ym.group(1) if ym else ""
        label = key
        if am:
            authors = am.group(1)
            first = authors.split(" and ")[0].strip()
            surname = first.split(",")[0].strip() if "," in first else first.split()[-1]
            multi = " and " in authors
            label = f"{surname} et al." if multi else surname
        out[key] = f"{label} {year}".strip()
    return out


def replace_cites(md: str, bib: dict[str, str]) -> str:
    def fmt(keys: str) -> list[str]:
        return [bib.get(k.strip(), k.strip()) for k in keys.split(",")]
    # \citep[pre][post]{keys} and \citep{keys}
    md = re.sub(r"\\citep(?:\[[^\]]*\])*\{([^}]*)\}",
                lambda m: "(" + "; ".join(fmt(m.group(1))) + ")", md)
    md = re.sub(r"\\citet(?:\[[^\]]*\])*\{([^}]*)\}",
                lambda m: ", ".join(fmt(m.group(1))), md)
    md = re.sub(r"\\citealt\{([^}]*)\}",
                lambda m: "; ".join(fmt(m.group(1))), md)
    return md


def link_callback(uri, rel):
    """Resolve figure/font URIs to absolute filesystem paths."""
    if os.path.isabs(uri) and os.path.exists(uri):
        return uri
    p = (HERE / uri).resolve()
    return str(p) if p.exists() else uri


CSS = """
@page { size: A4; margin: 1.8cm 1.9cm; }
body { font-family: "DV"; font-size: 9.6pt; line-height: 1.38; color: #111; text-align: justify; }
h1 { font-size: 17pt; line-height: 1.2; margin: 0 0 2pt 0; }
h2 { font-size: 12.5pt; margin: 14pt 0 4pt 0; border-bottom: 0.6pt solid #bbb; padding-bottom: 2pt; }
h3 { font-size: 10.6pt; margin: 9pt 0 3pt 0; }
p { margin: 0 0 6pt 0; }
code { font-family: "DV"; background: #f0f0f0; font-size: 8.8pt; }
blockquote { background: #eef4fb; border-left: 3pt solid #2c7fb8; margin: 6pt 0; padding: 5pt 9pt; font-size: 9.2pt; }
img { width: 86%; }
hr { border: 0; border-top: 0.5pt solid #ccc; }
table { border-collapse: collapse; font-size: 8.8pt; }
td, th { border: 0.5pt solid #aaa; padding: 2pt 5pt; }
"""


def register_fonts() -> None:
    """Register DejaVu with reportlab directly (avoids xhtml2pdf's temp-copy path,
    which fails under the sandbox). xhtml2pdf then picks up the registered family."""
    pdfmetrics.registerFont(TTFont("DV", str(FONTDIR / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DV-b", str(FONTDIR / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DV-i", str(FONTDIR / "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFont(TTFont("DV-bi", str(FONTDIR / "DejaVuSans-BoldOblique.ttf")))
    pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-b", italic="DV-i", boldItalic="DV-bi")


def main() -> int:
    register_fonts()
    bib = parse_bib(BIB)
    md_text = MD.read_text(encoding="utf-8")
    md_text = replace_cites(md_text, bib)
    body = markdown.markdown(md_text, extensions=["extra", "tables", "sane_lists", "smarty"])
    css = CSS
    html = f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"
    with open(OUT, "wb") as f:
        res = pisa.CreatePDF(html, dest=f, link_callback=link_callback, encoding="utf-8")
    if res.err:
        print(f"PDF generation had {res.err} error(s)")
        return 1
    print(f"wrote {OUT}  ({OUT.stat().st_size//1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
