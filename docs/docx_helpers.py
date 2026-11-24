"""python-docx үшін көмекші функциялар: МЕМСТ стиліндегі рәсімдеу (TNR 14, 1,5 интервал, 1,25 см азат жол)."""
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

FONT = "Times New Roman"


def set_font(run_or_style, size=14, bold=None, italic=None):
    f = run_or_style.font
    f.name = FONT; f.size = Pt(size)
    if bold is not None: f.bold = bold
    if italic is not None: f.italic = italic
    el = run_or_style.element if hasattr(run_or_style, "element") else run_or_style._element
    rpr = el.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts"); rpr.append(rfonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(a), FONT)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rfonts.get(qn(a)) is not None:
            del rfonts.attrib[qn(a)]


class Doc:
    def __init__(self):
        self.d = Document()
        s = self.d.sections[0]
        s.page_height, s.page_width = Cm(29.7), Cm(21.0)
        s.left_margin, s.right_margin, s.top_margin, s.bottom_margin = Cm(3), Cm(1.5), Cm(2), Cm(2)
        st = self.d.styles["Normal"]; set_font(st, 14)
        pf = st.paragraph_format
        pf.line_spacing = 1.5; pf.space_before = Pt(0); pf.space_after = Pt(0)
        pf.first_line_indent = Cm(1.25); pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for name, size in (("Heading 1", 14), ("Heading 2", 14), ("Heading 3", 14)):
            h = self.d.styles[name]; set_font(h, size, bold=True, italic=False)
            h.font.color.rgb = RGBColor(0, 0, 0)
            p = h.paragraph_format
            p.space_before = Pt(0); p.space_after = Pt(12); p.keep_with_next = True
            p.line_spacing = 1.5; p.first_line_indent = Cm(1.25); p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        self.d.styles["Heading 2"].paragraph_format.space_before = Pt(12)
        from docx.enum.style import WD_STYLE_TYPE
        for lvl, ind in ((1, 0), (2, 0.5)):
            name = f"toc {lvl}"
            try:
                st_ = self.d.styles[name]
            except KeyError:
                st_ = self.d.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
                st_.element.set(qn("w:styleId"), f"TOC{lvl}")
                if st_.element.get(qn("w:customStyle")) is not None:
                    del st_.element.attrib[qn("w:customStyle")]
            st_.base_style = self.d.styles["Normal"]
            set_font(st_, 14)
            q = st_.paragraph_format
            q.first_line_indent = Cm(0); q.left_indent = Cm(ind); q.alignment = WD_ALIGN_PARAGRAPH.LEFT
            q.space_after = Pt(0); q.line_spacing = 1.5
            q.tab_stops.add_tab_stop(Cm(16.5), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        self.fig_n = {}; self.tab_n = {}; self.eq_n = {}
        self.chapter = 0
        self._footer_page_numbers()

    # ---------- құрылым ----------
    def _footer_page_numbers(self):
        s = self.d.sections[0]
        s.different_first_page_header_footer = True
        p = s.footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        self._field(p, "PAGE", "1")

    def _field(self, p, code, placeholder=""):
        r = p.add_run(); set_font(r, 12)
        b = OxmlElement("w:fldChar"); b.set(qn("w:fldCharType"), "begin"); r._r.append(b)
        r2 = p.add_run(); set_font(r2, 12)
        it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = f" {code} "; r2._r.append(it)
        r3 = p.add_run(); set_font(r3, 12)
        sep = OxmlElement("w:fldChar"); sep.set(qn("w:fldCharType"), "separate"); r3._r.append(sep)
        r4 = p.add_run(placeholder); set_font(r4, 12)
        r5 = p.add_run(); e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), "end"); r5._r.append(e)

    def page_break(self):
        self.d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def toc(self):
        self.structural("МАЗМҰНЫ", in_toc=False)
        p = self.d.add_paragraph(); p.paragraph_format.first_line_indent = Cm(0)
        r = p.add_run(); b = OxmlElement("w:fldChar"); b.set(qn("w:fldCharType"), "begin"); r._r.append(b)
        r2 = p.add_run(); it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve")
        it.text = ' TOC \\o "1-2" \\h \\z \\u '; r2._r.append(it)
        r3 = p.add_run(); s = OxmlElement("w:fldChar"); s.set(qn("w:fldCharType"), "separate"); r3._r.append(s)
        p.add_run("Мазмұнды жаңарту үшін F9 басыңыз")
        r5 = p.add_run(); e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), "end"); r5._r.append(e)

    def structural(self, text, in_toc=True):
        """КІРІСПЕ, ҚОРЫТЫНДЫ сияқты құрылымдық элементтер: жаңа беттен, ортада, бас әріппен."""
        if in_toc:
            h = self.d.add_heading(text.upper(), level=1)
        else:
            h = self.d.add_paragraph(); r = h.add_run(text.upper()); set_font(r, 14, bold=True)
            h.paragraph_format.space_after = Pt(12)
        h.alignment = WD_ALIGN_PARAGRAPH.CENTER
        h.paragraph_format.first_line_indent = Cm(0)
        h.paragraph_format.page_break_before = True
        return h

    def h1(self, text):
        self.chapter += 1
        h = self.d.add_heading(f"{self.chapter} {text.upper()}", level=1)
        h.paragraph_format.page_break_before = True
        h.paragraph_format.left_indent = Cm(0)
        return h

    def h2(self, num, text):
        return self.d.add_heading(f"{num} {text}", level=2)

    # ---------- мәтін ----------
    def p(self, text, indent=True, align=None, bold=False, italic=False, size=14, space_after=0):
        para = self.d.add_paragraph()
        self._rich(para, text, bold, italic, size)
        if not indent: para.paragraph_format.first_line_indent = Cm(0)
        if align == "center": para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif align == "left": para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        elif align == "right": para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        if space_after: para.paragraph_format.space_after = Pt(space_after)
        return para

    def _rich(self, para, text, bold=False, italic=False, size=14):
        """**қалың**, _{төменгі индекс} және ^{жоғарғы индекс} белгілеулерін қолдайды."""
        import re
        for part in re.split(r"(\*\*.+?\*\*|_\{.+?\}|\^\{.+?\})", text):
            if not part: continue
            b, sub, sup = bold, False, False
            if part.startswith("**"): part, b = part[2:-2], True
            elif part.startswith("_{"): part, sub = part[2:-1], True
            elif part.startswith("^{"): part, sup = part[2:-1], True
            r = para.add_run(part); set_font(r, size, bold=b, italic=italic)
            if sub: r.font.subscript = True
            if sup: r.font.superscript = True

    def bullets(self, items, numbered=False):
        for k, it in enumerate(items, 1):
            mark = f"{k})" if numbered else "–"
            para = self.d.add_paragraph()
            self._rich(para, f"{mark} {it}")
        return

    def eq(self, formula, chap=None):
        chap = chap or self.chapter
        self.eq_n[chap] = self.eq_n.get(chap, 0) + 1
        para = self.d.add_paragraph(); para.paragraph_format.first_line_indent = Cm(0)
        para.paragraph_format.space_before = Pt(6); para.paragraph_format.space_after = Pt(6)
        ts = para.paragraph_format.tab_stops
        ts.add_tab_stop(Cm(8.25), WD_TAB_ALIGNMENT.CENTER); ts.add_tab_stop(Cm(16.5), WD_TAB_ALIGNMENT.RIGHT)
        r = para.add_run("\t"); set_font(r, 14)
        r = para.add_run(formula); set_font(r, 14)  # Word-та OMML теңдеуге айналдырылады (finalize_word.ps1)
        r = para.add_run(f"\t({chap}.{self.eq_n[chap]})"); set_font(r, 14)
        return f"{chap}.{self.eq_n[chap]}"

    def where(self, text):
        return self.p(text, indent=False)

    def figure(self, path, caption, width_cm=16.0, chap=None):
        chap = chap or self.chapter
        self.fig_n[chap] = self.fig_n.get(chap, 0) + 1
        n = f"{chap}.{self.fig_n[chap]}"
        para = self.d.add_paragraph(); para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.first_line_indent = Cm(0); para.paragraph_format.keep_with_next = True
        para.paragraph_format.space_before = Pt(6); para.paragraph_format.line_spacing = 1.0
        para.add_run().add_picture(path, width=Cm(width_cm))
        cap = self.p(f"{n}-сурет – {caption}", indent=False, align="center", space_after=6)
        return n

    def next_fig(self, chap=None):
        chap = chap or self.chapter
        return f"{chap}.{self.fig_n.get(chap, 0) + 1}"

    def next_tab(self, chap=None):
        chap = chap or self.chapter
        return f"{chap}.{self.tab_n.get(chap, 0) + 1}"

    def table(self, caption, header, rows, widths_cm, size=12, chap=None, align=None):
        n = None
        if caption:
            chap = chap or self.chapter
            self.tab_n[chap] = self.tab_n.get(chap, 0) + 1
            n = f"{chap}.{self.tab_n[chap]}"
            cp = self.p(f"{n}-кесте – {caption}", indent=False, align="left")
            cp.paragraph_format.keep_with_next = True; cp.paragraph_format.space_before = Pt(6)
        t = self.d.add_table(rows=1, cols=len(header)); t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, h in enumerate(header):
            self._cell(t.rows[0].cells[i], h, size, bold=True, center=True)
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                center = align[i] == "c" if align else i > 0
                self._cell(cells[i], str(v), size, center=center)
        for row in t.rows:
            for i, w in enumerate(widths_cm):
                row.cells[i].width = Cm(w)
        # тақырып жолын әр бетте қайталау
        trpr = t.rows[0]._tr.get_or_add_trPr(); th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true"); trpr.append(th)
        sp = self.d.add_paragraph(); sp.paragraph_format.space_after = Pt(6)
        return n

    def _cell(self, cell, text, size, bold=False, center=False):
        para = cell.paragraphs[0]
        para.paragraph_format.first_line_indent = Cm(0); para.paragraph_format.line_spacing = 1.0
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
        self._rich(para, text, bold=bold, size=size)

    def code(self, text, size=10):
        for line in text.rstrip().split("\n"):
            para = self.d.add_paragraph(); pf = para.paragraph_format
            pf.first_line_indent = Cm(0); pf.line_spacing = 1.0; pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = para.add_run(line if line else " "); r.font.size = Pt(size); r.font.name = "Consolas"
            rpr = r._r.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
            if rf is None: rf = OxmlElement("w:rFonts"); rpr.append(rf)
            for a in ("w:ascii", "w:hAnsi", "w:cs"): rf.set(qn(a), "Consolas")

    def save(self, path):
        self.d.save(path)
