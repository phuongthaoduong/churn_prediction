"""Convert weekly_report_step2.md to a professional .docx with Times New Roman, compact spacing."""

import re
from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml

import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
INPUT = os.path.join(_PROJECT_ROOT, "docs", "weekly_report_step2.md")
OUTPUT = os.path.join(_PROJECT_ROOT, "profiling_output", "weekly_report_step2_cleaning.docx")

FONT_NAME = "Times New Roman"
FONT_SIZE_BODY = Pt(11)
FONT_SIZE_TABLE = Pt(9)
FONT_SIZE_H1 = Pt(14)
FONT_SIZE_H2 = Pt(12)
FONT_SIZE_H3 = Pt(11)
FONT_SIZE_META = Pt(10)

HEADER_BG = "1F4E79"  # dark blue
HEADER_FG = RGBColor(0xFF, 0xFF, 0xFF)
ALT_ROW_BG = "F2F2F2"


def set_cell_shading(cell, color_hex):
    tc = cell._element
    tcPr = tc.get_or_add_tcPr()
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}" w:val="clear"/>')
    tcPr.append(shading)


def set_paragraph_spacing(paragraph, before=0, after=0, line=240):
    """Set spacing in twips. 240 twips = single space, 276 = 1.15, 360 = 1.5."""
    pf = paragraph.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    spacing = paragraph._element.get_or_add_pPr().get_or_add_spacing()
    spacing.set(qn("w:line"), str(line))
    spacing.set(qn("w:lineRule"), "auto")


def apply_font(run, size=FONT_SIZE_BODY, bold=False, italic=False, color=None):
    run.font.name = FONT_NAME
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_NAME)
    run.font.size = size
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = color


def add_paragraph(doc, text, size=FONT_SIZE_BODY, bold=False, italic=False,
                  alignment=None, before=1, after=1, line=252, color=None):
    p = doc.add_paragraph()
    if alignment:
        p.alignment = alignment
    set_paragraph_spacing(p, before=before, after=after, line=line)

    # Handle inline bold (**text**) and italic (*text*)
    parts = re.split(r'(\*\*.*?\*\*|\*.*?\*)', text)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            run = p.add_run(part[2:-2])
            apply_font(run, size=size, bold=True, color=color)
        elif part.startswith("*") and part.endswith("*") and not part.startswith("**"):
            run = p.add_run(part[1:-1])
            apply_font(run, size=size, italic=True, color=color)
        else:
            run = p.add_run(part)
            apply_font(run, size=size, bold=bold, italic=italic, color=color)
    return p


def add_heading(doc, text, level=1):
    sizes = {1: FONT_SIZE_H1, 2: FONT_SIZE_H2, 3: FONT_SIZE_H3}
    befores = {1: 8, 2: 6, 3: 4}
    afters = {1: 3, 2: 2, 3: 2}

    p = doc.add_paragraph()
    set_paragraph_spacing(p, before=befores[level], after=afters[level], line=252)
    run = p.add_run(text)
    apply_font(run, size=sizes[level], bold=True,
               color=RGBColor(0x1F, 0x4E, 0x79) if level <= 2 else None)
    return p


def add_table(doc, rows):
    """Add a formatted table from parsed markdown rows."""
    if len(rows) < 2:
        return
    # Filter out separator rows (|---|---|)
    data_rows = [r for r in rows if not re.match(r'^\|[\s\-:|]+\|$', r)]
    if len(data_rows) < 2:
        return

    parsed = []
    for row_str in data_rows:
        cells = [c.strip() for c in row_str.strip('|').split('|')]
        parsed.append(cells)

    n_cols = len(parsed[0])
    table = doc.add_table(rows=len(parsed), cols=n_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    # Style
    tbl = table._element
    tblPr = tbl.tblPr if tbl.tblPr is not None else parse_xml(f'<w:tblPr {nsdecls("w")}/>')
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '  <w:top w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
        '  <w:left w:val="none"/>'
        '  <w:bottom w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
        '  <w:right w:val="none"/>'
        '  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="D9D9D9"/>'
        '  <w:insideV w:val="none"/>'
        '</w:tblBorders>'
    )
    tblPr.append(borders)

    for i, row_data in enumerate(parsed):
        row = table.rows[i]
        for j, cell_text in enumerate(row_data):
            if j >= n_cols:
                break
            cell = row.cells[j]
            # Clear default paragraph
            cell.paragraphs[0].clear()
            p = cell.paragraphs[0]
            set_paragraph_spacing(p, before=1, after=1, line=240)

            # Handle inline bold
            parts = re.split(r'(\*\*.*?\*\*)', cell_text)
            for part in parts:
                if part.startswith("**") and part.endswith("**"):
                    run = p.add_run(part[2:-2])
                    apply_font(run, size=FONT_SIZE_TABLE, bold=True)
                else:
                    run = p.add_run(part)
                    apply_font(run, size=FONT_SIZE_TABLE, bold=(i == 0))

            # Header row styling
            if i == 0:
                set_cell_shading(cell, HEADER_BG)
                for run in p.runs:
                    run.font.color.rgb = HEADER_FG
            elif i % 2 == 0:
                set_cell_shading(cell, ALT_ROW_BG)

    # Compact cell margins
    tbl_pr = table._element.tblPr
    margins = parse_xml(
        f'<w:tblCellMar {nsdecls("w")}>'
        '  <w:top w:w="30" w:type="dxa"/>'
        '  <w:left w:w="60" w:type="dxa"/>'
        '  <w:bottom w:w="30" w:type="dxa"/>'
        '  <w:right w:w="60" w:type="dxa"/>'
        '</w:tblCellMar>'
    )
    tbl_pr.append(margins)

    # Add small space after table
    after_p = doc.add_paragraph()
    set_paragraph_spacing(after_p, before=0, after=2, line=240)


def add_figure_placeholder(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_paragraph_spacing(p, before=3, after=3, line=240)
    run = p.add_run(text)
    apply_font(run, size=Pt(9), italic=True, color=RGBColor(0x60, 0x60, 0x60))


def add_horizontal_rule(doc):
    p = doc.add_paragraph()
    set_paragraph_spacing(p, before=2, after=2, line=240)
    pPr = p._element.get_or_add_pPr()
    pBdr = parse_xml(
        f'<w:pBdr {nsdecls("w")}>'
        '  <w:bottom w:val="single" w:sz="4" w:space="1" w:color="BFBFBF"/>'
        '</w:pBdr>'
    )
    pPr.append(pBdr)


def convert():
    with open(INPUT, "r", encoding="utf-8") as f:
        lines = f.readlines()

    doc = Document()

    # Page margins: narrow
    for section in doc.sections:
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    # Remove default empty paragraph
    if doc.paragraphs:
        doc.paragraphs[0]._element.getparent().remove(doc.paragraphs[0]._element)

    i = 0
    table_buffer = []
    in_table = False

    while i < len(lines):
        line = lines[i].rstrip('\n')

        # Skip empty lines unless we need to flush table
        if not line.strip():
            if in_table and table_buffer:
                add_table(doc, table_buffer)
                table_buffer = []
                in_table = False
            i += 1
            continue

        # Horizontal rule
        if line.strip() == '---':
            if in_table and table_buffer:
                add_table(doc, table_buffer)
                table_buffer = []
                in_table = False
            add_horizontal_rule(doc)
            i += 1
            continue

        # Table rows
        if line.strip().startswith('|'):
            in_table = True
            table_buffer.append(line.strip())
            i += 1
            continue
        else:
            if in_table and table_buffer:
                add_table(doc, table_buffer)
                table_buffer = []
                in_table = False

        # Title (# heading)
        if line.startswith('# ') and not line.startswith('## '):
            text = line[2:].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_paragraph_spacing(p, before=0, after=6, line=252)
            run = p.add_run(text)
            apply_font(run, size=Pt(16), bold=True, color=RGBColor(0x1F, 0x4E, 0x79))
            i += 1
            continue

        # Metadata block (> **Key:** value)
        if line.startswith('> '):
            text = line[2:].strip()
            add_paragraph(doc, text, size=FONT_SIZE_META, before=0, after=0, line=240,
                         color=RGBColor(0x40, 0x40, 0x40))
            i += 1
            continue

        # H2
        if line.startswith('## '):
            text = line[3:].strip()
            add_heading(doc, text, level=2)
            i += 1
            continue

        # H3
        if line.startswith('### '):
            text = line[4:].strip()
            add_heading(doc, text, level=3)
            i += 1
            continue

        # Figure placeholder
        if line.strip().startswith('![]('):
            # Combine with caption on same or next line
            full = line.strip()
            if i + 1 < len(lines) and lines[i+1].strip().startswith('*Figure'):
                full = full + ' ' + lines[i+1].strip()
                i += 1
            # Extract caption
            caption_match = re.search(r'\*(.+?)\*', full)
            if caption_match:
                add_figure_placeholder(doc, f"[{caption_match.group(1)}]")
            else:
                add_figure_placeholder(doc, "[Figure placeholder]")
            i += 1
            continue

        # Bullet points
        if line.strip().startswith('- '):
            text = line.strip()[2:]
            p = doc.add_paragraph()
            p.style = doc.styles['List Bullet'] if 'List Bullet' in [s.name for s in doc.styles] else None
            p.paragraph_format.left_indent = Cm(0.8)
            set_paragraph_spacing(p, before=0, after=0, line=252)
            parts = re.split(r'(\*\*.*?\*\*|\*.*?\*)', text)
            for part in parts:
                if part.startswith("**") and part.endswith("**"):
                    run = p.add_run(part[2:-2])
                    apply_font(run, size=FONT_SIZE_BODY, bold=True)
                elif part.startswith("*") and part.endswith("*"):
                    run = p.add_run(part[1:-1])
                    apply_font(run, size=FONT_SIZE_BODY, italic=True)
                else:
                    run = p.add_run(part)
                    apply_font(run, size=FONT_SIZE_BODY)
            i += 1
            continue

        # Regular paragraph
        text = line.strip()
        add_paragraph(doc, text, size=FONT_SIZE_BODY, before=2, after=2, line=252)
        i += 1

    # Flush remaining table
    if in_table and table_buffer:
        add_table(doc, table_buffer)

    doc.save(OUTPUT)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    convert()
