# -*- coding: utf-8 -*-
"""论文 markdown → 排版 HTML → Edge 无头渲染 PDF

用法：
  python scripts/paper_figs/md_to_pdf.py docs/paper_draft.md docs/paper_en.pdf
  python scripts/paper_figs/md_to_pdf.py docs/paper_draft_cn.md docs/paper_cn.pdf

用 Edge 无头模式渲染（Windows 自带，支持中文字体）。
"""
import re
import subprocess
import sys
from pathlib import Path

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

HTML_TMPL = """<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  @page {{ size: A4; margin: 2cm 2.5cm; }}
  @font-face {{ font-family: "CJK"; src: url("file:///C:/Windows/Fonts/msyh.ttc"); }}
  body {{ font-family: {font}; font-size: 11pt; line-height: 1.6; color: #111; max-width: 800px; margin: 0 auto; padding: 0 20px; }}
  h1 {{ font-size: 18pt; text-align: center; margin-top: 0.5em; }}
  h2 {{ font-size: 14pt; margin-top: 1.2em; border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }}
  h3 {{ font-size: 12pt; margin-top: 1em; }}
  p {{ text-align: justify; margin: 0.6em 0; }}
  img {{ max-width: 100%; display: block; margin: 1em auto; }}
  .figure {{ text-align: center; font-size: 9.5pt; color: #333; margin: 0.8em 0 1.5em; }}
  table {{ border-collapse: collapse; margin: 1em auto; }}
  th, td {{ border: 1px solid #999; padding: 4px 8px; font-size: 10pt; }}
  th {{ background: #f0f0f0; }}
  li {{ margin: 0.3em 0; }}
  blockquote {{ border-left: 3px solid #ccc; margin: 1em 0; padding-left: 1em; color: #555; }}
  code {{ background: #f5f5f5; padding: 1px 4px; font-size: 9.5pt; }}
  .refs {{ font-size: 10pt; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def inline(text: str) -> str:
    """处理行内语法：加粗、斜体、代码、链接。"""
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', text)
    return text


def md_to_html(md: str) -> str:
    """轻量 markdown → HTML（覆盖论文用到的语法）。"""
    lines = md.split("\n")
    html = []
    i = 0
    in_ul = False
    in_table = False
    table_rows = []

    def close_ul():
        nonlocal in_ul
        if in_ul:
            html.append("</ul>")
            in_ul = False

    def close_table():
        nonlocal in_table
        if in_table:
            html.append("</table>")
            in_table = False

    while i < len(lines):
        line = lines[i].rstrip()

        # 标题
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            close_ul(); close_table()
            lvl = len(m.group(1))
            html.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
            i += 1
            continue

        # 图 ![alt](path)
        m = re.match(r"!\[(.*?)\]\((.*?)\)", line)
        if m:
            close_ul(); close_table()
            html.append(f'<img src="{m.group(2)}" alt="{m.group(1)}">')
            i += 1
            # 图注（紧跟的 **Figure N.** 行）
            if i < len(lines) and lines[i].strip().startswith("**Figure"):
                html.append(f'<div class="figure">{inline(lines[i].strip())}</div>')
                i += 1
            continue

        # 表格
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not in_table:
                in_table = True
                html.append("<table>")
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                i += 1
                continue
            html.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
            i += 1
            continue
        else:
            if in_table:
                close_table()

        # 列表
        m = re.match(r"^\s*[-*]\s+(.*)", line)
        if m:
            if not in_ul:
                html.append("<ul>")
                in_ul = True
            html.append(f"<li>{inline(m.group(1))}</li>")
            i += 1
            continue
        else:
            close_ul()

        # 有序列表
        m = re.match(r"^\s*\d+\.\s+(.*)", line)
        if m:
            close_ul()
            html.append(f"<p>{inline(m.group(1))}</p>")
            i += 1
            continue

        # 分隔线
        if re.match(r"^\s*-{3,}\s*$", line) or re.match(r"^\s*\*{3,}\s*$", line):
            close_ul(); close_table()
            html.append("<hr>")
            i += 1
            continue

        # 空行
        if not line.strip():
            close_ul()
            i += 1
            continue

        # 普通段落（含引用）
        close_ul()
        if line.startswith(">"):
            html.append(f"<blockquote>{inline(line.lstrip('>').strip())}</blockquote>")
        else:
            html.append(f"<p>{inline(line)}</p>")
        i += 1

    close_ul(); close_table()
    return "\n".join(html)


def main():
    md_path = Path(sys.argv[1])
    pdf_path = Path(sys.argv[2])
    md = md_path.read_text(encoding="utf-8")

    # 判断语言：中文论文用中文字体，英文用 serif
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in md[:500])
    if has_cjk:
        font = "CJK, SimSun, serif"
        lang = "zh"
    else:
        font = "Times New Roman, serif"
        lang = "en"

    # 标题 = 第一个 # 行
    title = "RAMS"
    for line in md.split("\n"):
        if line.startswith("# "):
            title = line[2:].strip()
            break

    body = md_to_html(md)

    # 图片路径相对 docs/ 解析
    import os
    base_dir = md_path.parent
    body = re.sub(r'src="(paper_figs/[^"]+)"',
                  lambda m: f'src="{(base_dir / m.group(1)).resolve().as_posix()}"',
                  body)

    html = HTML_TMPL.format(lang=lang, title=title, font=font, body=body)
    html_path = md_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")

    # Edge 无头打印 PDF
    cmd = [EDGE, "--headless", "--disable-gpu", "--no-sandbox",
           f"--print-to-pdf={pdf_path.resolve()}", f"file:///{html_path.resolve().as_posix()}"]
    r = subprocess.run(cmd, capture_output=True, timeout=120)
    if r.returncode != 0:
        print(f"Edge 渲染失败: {r.stderr.decode(errors='replace')}")
        sys.exit(1)
    print(f"PDF 已生成: {pdf_path} ({pdf_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
