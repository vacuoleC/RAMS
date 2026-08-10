# -*- coding: utf-8 -*-
"""论文 markdown → PDF（reportlab 纯 Python，中文字体用 ttf）

用法：
  python scripts/paper_figs/md_to_pdf_rl.py docs/paper_draft.md docs/paper_en.pdf
  python scripts/paper_figs/md_to_pdf_rl.py docs/paper_draft_cn.md docs/paper_cn.pdf

不依赖系统渲染（reportlab 直接画 PDF），中文用 Windows ttf 字体。
"""
import re
import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table

# 注册中文字体（优先单文件 ttf，最稳）
_CJK_FONT = None
for _path in [r"C:/Windows/Fonts/simhei.ttf", r"C:/Windows/Fonts/Deng.ttf",
              r"C:/Windows/Fonts/simfang.ttf", r"C:/Windows/Fonts/simsun.ttc",
              r"C:/Windows/Fonts/msyh.ttc"]:
    try:
        pdfmetrics.registerFont(TTFont("CJK", _path))
        _CJK_FONT = "CJK"
        print(f"注册中文字体: {_path}", file=sys.stderr)
        break
    except Exception as _e:
        print(f"字体注册失败 {_path}: {_e}", file=sys.stderr)
        continue

_CN_FONT = "CJK" if _CJK_FONT else "Helvetica"


def _make_styles(font):
    return {
        "h1": ParagraphStyle("h1", fontName=font, fontSize=16, leading=22,
                             alignment=1, spaceAfter=10),
        "h2": ParagraphStyle("h2", fontName=font, fontSize=13, leading=18,
                             spaceBefore=12, spaceAfter=6),
        "h3": ParagraphStyle("h3", fontName=font, fontSize=11, leading=16,
                             spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("body", fontName=font, fontSize=10, leading=15,
                               alignment=4, spaceAfter=6),
        "caption": ParagraphStyle("caption", fontName=font, fontSize=8.5, leading=12,
                                  alignment=1, spaceAfter=10, textColor="#333"),
        "table": ParagraphStyle("table", fontName=font, fontSize=8.5, leading=12),
    }


def _inline(text):
    """把 markdown 行内标记转成 reportlab 支持的 <b>/<i>/<code>。"""
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    return text


def _parse(md_path, styles, base_dir):
    lines = Path(md_path).read_text(encoding="utf-8").split("\n")
    flow = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            lvl = len(m.group(1))
            flow.append(Paragraph(_inline(m.group(2)), styles[f"h{lvl}"]))
            i += 1
            continue
        m = re.match(r"!\[(.*?)\]\((.*?)\)", line)
        if m:
            img = base_dir / m.group(2)
            if img.exists():
                flow.append(Image(str(img), width=15*cm, height=15*cm*0.66,
                                  hAlign="CENTER"))
                # 图注
                if i+1 < len(lines) and lines[i+1].strip().startswith("**"):
                    flow.append(Paragraph(_inline(lines[i+1].strip()), styles["caption"]))
                    i += 2
                    continue
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append([Paragraph(_inline(c), styles["table"]) for c in cells])
                i += 1
            if rows:
                flow.append(Table(rows, hAlign="CENTER"))
            continue
        if re.match(r"^\s*[-*]\s+", line):
            m = re.match(r"^\s*[-*]\s+(.*)", line)
            flow.append(Paragraph("• " + _inline(m.group(1)), styles["body"]))
            i += 1
            continue
        if re.match(r"^\s*[-*]{3,}\s*$", line):
            flow.append(Spacer(1, 8))
            i += 1
            continue
        if not line.strip():
            i += 1
            continue
        flow.append(Paragraph(_inline(line), styles["body"]))
        i += 1
    return flow


def main():
    md_path = Path(sys.argv[1])
    pdf_path = Path(sys.argv[2])
    md_text = md_path.read_text(encoding="utf-8")
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in md_text[:500])
    font = _CN_FONT if has_cjk else "Helvetica"
    styles = _make_styles(font)

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                            leftMargin=2.5*cm, rightMargin=2.5*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    flow = _parse(md_path, styles, md_path.parent)
    doc.build(flow)
    print(f"PDF 已生成: {pdf_path} ({pdf_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
