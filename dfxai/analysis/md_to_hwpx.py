"""
논문 원고(markdown)를 팀 원본 hwpx 의 서식에 부어 넣어 새 hwpx 를 만든다.

    python -m dfxai.analysis.md_to_hwpx --md paper/본논문_재작성.md --template <원본.hwpx> --out <새.hwpx> \
        --pic "<그림 4-1>=1" --new-pic "<그림 4-2>=results/paper_figs_rewrite/fig4_3_tactics.png:39118"

원본 문단(장·절·항·본문·표 제목·註·그림·각주·참고문헌)을 틀로 복제하므로 [붙임 3] 양식의 글꼴과 문단
모양이 유지된다. 원고 형식:
  * '# 제목', '### - 부제 -', '(영문 제목)', '## - 요 약 문 -' 다음 문단, '**주제어**: …'
  * 본문 '## n. 장', '### n.n. 절', '#### n.n.n. 항', 문단 끝의 각주 표시 'n)'(1부터 차례로)
  * '<표 n-n> 제목' 다음 줄부터 마크다운 표, '註:'·'出處:' 줄
  * '*[<그림 n-n> 제목]*' 줄 다음 '註:'·'出處:' 줄
  * '**각주**' 다음 'n) 내용'(기울임은 *…*), '## - 참 고 문 헌 -' 다음 항목, '**영문 제목**', '## - Abstract -', '**Keywords**: …'
--pic 은 원본 hwpx 의 그림(등장 순서 0부터)을 새 그림 번호에 다시 쓰고, --new-pic 은 PNG 를 새로 넣는다(폭 HWPUNIT, 본문 폭 39118).
"""
from __future__ import annotations
import argparse, copy, io, os, re, zipfile
from lxml import etree
from PIL import Image

HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"
def q(t): return "{%s}%s" % (HP, t)
OLD_TITLE_KO = "공중전 인공지능 모델의 설명가능성과 전투성능 트레이드오프 분석"
OLD_TITLE_EN = ("Explainability–Performance Trade-off in Combat AI: A Comparison of Rule-based, "
                "Learning-based, and Hybrid Models in a Simulated 1v1 Dogfight")


def clean(s: str) -> str:
    return s.replace("\\~", "~").replace("\\*", "*").replace("**", "")


def strip_layout(el):
    for ls in list(el.iter(q("linesegarray"))):
        ls.getparent().remove(ls)


def ptext(p): return "".join(t.text or "" for t in p.iter(q("t")))


def set_single_text(p, text):
    """문단의 첫 글자 run 에 text 를 넣는다. '<br>' 은 한글의 줄바꿈(hp:lineBreak)으로 바꾼다."""
    runs = [r for r in p.findall(q("run")) if r.find(q("t")) is not None]
    for r in runs:
        for t in r.findall(q("t")):
            t.text = ""
            for c in list(t):
                t.remove(c)
    t = runs[0].find(q("t"))
    parts = text.split("<br>")
    t.text = parts[0]
    for part in parts[1:]:
        lb = etree.SubElement(t, q("lineBreak"))
        lb.tail = part
    strip_layout(p)


def parse_md(md: str) -> dict:
    L = md.split("\n")
    def idx(prefix, start=0):
        for i in range(start, len(L)):
            if L[i].startswith(prefix):
                return i
        raise KeyError(prefix)
    d = {}
    d["title"] = clean(L[idx("# ")][2:].strip())
    d["subtitle"] = clean(L[idx("### - ")][4:].strip())
    d["eng_title_paren"] = clean(next(x for x in L if x.startswith("(") and x.rstrip().endswith(")")).strip())
    i_ab, i_kw = idx("## - 요 약 문 -"), idx("**주제어**")
    d["abstract_ko"] = clean(" ".join(x.strip() for x in L[i_ab + 1:i_kw] if x.strip() and x.strip() != "---"))
    d["keywords_ko"] = "주제어 : " + clean(L[i_kw].split(":", 1)[1].strip())
    i_body, i_fn, i_ref = idx("## 1. "), idx("**각주**"), idx("## - 참 고 문 헌 -")
    d["body"] = L[i_body:i_fn]
    d["footnotes"] = {int(m.group(1)): m.group(2) for x in L[i_fn + 1:i_ref]
                      if (m := re.match(r"^(\d+)\) (.*)$", x.strip()))}
    i_eng = next(i for i in range(i_ref, len(L)) if L[i].startswith("**") and not L[i].startswith("**국문")
                 and not L[i].startswith("**영문") and not L[i].startswith("**Keywords"))
    d["refs"] = [x for x in L[i_ref + 1:i_eng] if x.strip() and not x.startswith("**") and x.strip() != "---"]
    d["eng_title"] = clean(L[i_eng].strip())
    i_abs, i_kwe = idx("## - Abstract -"), idx("**Keywords**")
    d["abstract_en"] = clean(" ".join(x.strip() for x in L[i_abs + 1:i_kwe] if x.strip()))
    d["keywords_en"] = "Keywords : " + clean(L[i_kwe].split(":", 1)[1].strip())
    return d


def parse_blocks(B: list[str]) -> list:
    blocks, k = [], 0
    while k < len(B):
        ln = B[k].rstrip()
        nxt = next((B[j] for j in range(k + 1, len(B)) if B[j].strip()), "")
        if not ln.strip() or ln.strip() == "---":
            k += 1; continue
        if ln.startswith("#### "):
            blocks.append(("SUB", " " + clean(ln[5:])))
        elif ln.startswith("### "):
            blocks.append(("SEC", clean(ln[4:])))
        elif ln.startswith("## "):
            blocks.append(("CH", clean(ln[3:])))
        elif ln.startswith("<표 ") and nxt.startswith("|"):
            blocks.append(("TCAP", clean(ln)))
            rows = []; k += 1
            while k < len(B) and (B[k].startswith("|") or not B[k].strip()):
                if B[k].startswith("|") and not re.match(r"^\|[-| ]+\|$", B[k]):
                    rows.append([clean(c.strip()) for c in B[k].strip().strip("|").split("|")])
                if not B[k].strip() and rows:
                    break
                k += 1
            blocks.append(("TABLE", rows)); continue
        elif ln.startswith("*[<그림"):
            m = re.match(r"^\*\[(<그림 [\d-]+>) (.*)\]\*$", ln.strip())
            blocks.append(("FIG", m.group(1), clean(m.group(2))))
        elif ln.startswith("註:") or ln.startswith("出處:"):
            blocks.append(("NOTE", clean(ln)))
        else:
            blocks.append(("P", clean(ln)))
        k += 1
    return blocks


def build(md_path, template, out, pic_map: dict, new_pics: dict):
    d = parse_md(io.open(md_path, encoding="utf-8").read())
    blocks = parse_blocks(d["body"])
    footnotes = d["footnotes"]; n_fn = max(footnotes) if footnotes else 0
    zin = zipfile.ZipFile(template)
    r0 = etree.fromstring(zin.read("Contents/section0.xml"))
    r1 = etree.fromstring(zin.read("Contents/section1.xml"))
    r2 = etree.fromstring(zin.read("Contents/section2.xml"))

    def find_p(root, pred):
        return next(p for p in root if pred(p))
    T_CH_FIRST = r1[0]
    T_CH = find_p(r1, lambda p: p.get("styleIDRef") == "9" and p.find(".//" + q("secPr")) is None)
    T_SEC = find_p(r1, lambda p: p.get("styleIDRef") == "10")
    T_SUB = find_p(r1, lambda p: p.get("styleIDRef") == "11")
    T_BODY = find_p(r1, lambda p: p.get("paraPrIDRef") == "41" and ptext(p).strip()
                    and p.find(".//" + q("footNote")) is None and len(p.findall(q("run"))) == 1)
    T_BLANK = find_p(r1, lambda p: p.get("paraPrIDRef") == "41" and not ptext(p).strip()
                     and p.find(".//" + q("tbl")) is None)
    T_TCAP = find_p(r1, lambda p: p.get("paraPrIDRef") == "44" and ptext(p).startswith("<표"))
    T_FCAP = find_p(r1, lambda p: p.get("paraPrIDRef") == "1" and ptext(p).startswith("<그림"))
    T_NOTE = find_p(r1, lambda p: p.get("styleIDRef") == "15")
    T_TBLP = next(p for p in r1 if p.find(".//" + q("tbl")) is not None
                  and int(p.find(".//" + q("tbl")).get("colCnt")) >= 6)
    PICS = [p for p in r1 if p.find(".//" + q("pic")) is not None]
    FN_T = copy.deepcopy(next(r1.iter(q("footNote"))))

    counter = [0]
    def make_fn(num, text):
        fn = copy.deepcopy(FN_T)
        fn.set("number", str(num)); fn.set("instId", str(1300000000 + num))
        p = fn.find(".//" + q("p")); strip_layout(p)
        for r in p.findall(q("run")):
            p.remove(r)
        ctrl = copy.deepcopy(FN_T.find(".//" + q("autoNum")).getparent())
        ctrl.find(q("autoNum")).set("num", str(num))
        parts = re.split(r"\*([^*]+)\*", text)
        r = etree.SubElement(p, q("run")); r.set("charPrIDRef", "17"); r.append(ctrl)
        etree.SubElement(r, q("t")).text = " " + clean(parts[0])
        for j in range(1, len(parts), 2):
            ri = etree.SubElement(p, q("run")); ri.set("charPrIDRef", "56")
            etree.SubElement(ri, q("t")).text = clean(parts[j])
            rp = etree.SubElement(p, q("run")); rp.set("charPrIDRef", "17")
            etree.SubElement(rp, q("t")).text = clean(parts[j + 1])
        return fn

    def make_body(text):
        p = copy.deepcopy(T_BODY); strip_layout(p)
        for r in p.findall(q("run")):
            p.remove(r)
        run = etree.SubElement(p, q("run")); run.set("charPrIDRef", "30")
        buf, i = [], 0
        while i < len(text):
            n = counter[0] + 1; mk = f"{n})"
            if (n <= n_fn and text.startswith(mk, i) and i > 0
                    and not text[i - 1].isdigit() and text[i - 1] not in " (."):
                etree.SubElement(run, q("t")).text = "".join(buf); buf = []
                c = etree.SubElement(run, q("ctrl")); c.append(make_fn(n, footnotes[n]))
                counter[0] = n; i += len(mk); continue
            if (n <= n_fn and text.startswith(mk, i) and i > 0 and text[i - 1] == "."
                    and (i < 2 or not text[i - 2].isdigit())):
                etree.SubElement(run, q("t")).text = "".join(buf); buf = []
                c = etree.SubElement(run, q("ctrl")); c.append(make_fn(n, footnotes[n]))
                counter[0] = n; i += len(mk); continue
            buf.append(text[i]); i += 1
        etree.SubElement(run, q("t")).text = "".join(buf)
        return p

    def clone_text(tmpl, text):
        p = copy.deepcopy(tmpl); set_single_text(p, text); return p

    tbl_seq = [0]
    def make_table(rows):
        p = copy.deepcopy(T_TBLP); strip_layout(p)
        tbl = p.find(".//" + q("tbl")); trs = tbl.findall(q("tr"))
        hdr_cell, body_cell = copy.deepcopy(trs[0][0]), copy.deepcopy(trs[1][1])
        for tr in trs:
            tbl.remove(tr)
        ncol = max(len(r) for r in rows)
        rows = [r + [""] * (ncol - len(r)) for r in rows]
        W = int(tbl.find(q("sz")).get("width"))
        def vis(s): return sum(2 if ord(ch) > 0x2E80 else 1 for ch in s)
        weights = []
        for c in range(ncol):
            body = max([vis(r[c]) for r in rows[1:]] or [0])
            weights.append(min(max(max(body, vis(rows[0][c]) * 0.7), 10), 46))
        widths = [int(W * w / sum(weights)) for w in weights]; widths[-1] += W - sum(widths)
        for ri, row in enumerate(rows):
            tr = etree.SubElement(tbl, q("tr"))
            for ci, txt in enumerate(row):
                tc = copy.deepcopy(hdr_cell if ri == 0 else body_cell); strip_layout(tc)
                set_single_text(tc.find(".//" + q("p")), txt)
                tc.set("header", "1" if ri == 0 else "0")
                tc.find(q("cellAddr")).set("colAddr", str(ci)); tc.find(q("cellAddr")).set("rowAddr", str(ri))
                tc.find(q("cellSpan")).set("colSpan", "1"); tc.find(q("cellSpan")).set("rowSpan", "1")
                tc.find(q("cellSz")).set("width", str(widths[ci])); tc.find(q("cellSz")).set("height", "1000")
                tr.append(tc)
        tbl.set("rowCnt", str(len(rows))); tbl.set("colCnt", str(ncol))
        tbl_seq[0] += 1
        tbl.set("id", str(1500000100 + tbl_seq[0])); tbl.set("zOrder", str(400 + tbl_seq[0]))
        tbl.find(q("sz")).set("height", str(1000 * len(rows)))
        return p

    new_items = []
    def make_pic(label):
        if label in pic_map:
            return copy.deepcopy(PICS[pic_map[label]])
        path, W = new_pics[label]
        w, h = Image.open(path).size
        item_id = f"image{len(PICS) + len(new_items) + 1}"
        new_items.append((item_id, path))
        p = copy.deepcopy(PICS[0]); strip_layout(p)
        pic = p.find(".//" + q("pic")); H = int(W * h / w)
        n = len(new_items)
        pic.set("id", str(1211997300 + n)); pic.set("instid", str(138255500 + n)); pic.set("zOrder", str(300 + n))
        for el in pic.iter():
            name = etree.QName(el).localname
            if name in ("orgSz", "sz"):
                el.set("width", str(W)); el.set("height", str(H))
            elif name == "rotationInfo":
                el.set("centerX", str(W // 2)); el.set("centerY", str(H // 2))
            elif name == "pt1":
                el.set("x", str(W)); el.set("y", "0")
            elif name == "pt2":
                el.set("x", str(W)); el.set("y", str(H))
            elif name == "pt3":
                el.set("x", "0"); el.set("y", str(H))
            elif name == "imgClip":
                el.set("right", str(w * 75)); el.set("bottom", str(h * 75))
            elif name == "imgDim":
                el.set("dimwidth", str(w * 75)); el.set("dimheight", str(h * 75))
            elif name == "img":
                el.set("binaryItemIDRef", item_id)
            elif name == "shapeComment":
                el.text = f"그림입니다.\n원본 그림의 이름: {os.path.basename(path)}\n원본 그림의 크기: 가로 {w}pixel, 세로 {h}pixel"
        return p

    out_ps, first_ch, prev = [], True, None
    def emit_blank():
        last = out_ps[-1] if out_ps else None
        if last is not None and not ptext(last).strip() and last.find(".//" + q("tbl")) is None \
                and last.find(".//" + q("pic")) is None:
            return
        out_ps.append(copy.deepcopy(T_BLANK))
    for b in blocks:
        kind = b[0]
        if kind == "CH":
            if first_ch:
                p = copy.deepcopy(T_CH_FIRST); set_single_text(p, b[1]); first_ch = False
            else:
                emit_blank(); p = clone_text(T_CH, b[1])
            out_ps.append(p)
        elif kind == "SEC":
            if prev != "CH":
                emit_blank()
            out_ps.append(clone_text(T_SEC, b[1]))
        elif kind == "SUB":
            out_ps.append(clone_text(T_SUB, b[1]))
        elif kind == "P":
            if prev == "NOTE":
                emit_blank()
            out_ps.append(make_body(b[1]))
        elif kind == "TCAP":
            emit_blank(); out_ps.append(clone_text(T_TCAP, b[1]))
        elif kind == "TABLE":
            out_ps.append(make_table(b[1]))
        elif kind == "FIG":
            emit_blank(); out_ps.append(make_pic(b[1])); out_ps.append(clone_text(T_FCAP, f"{b[1]} {b[2]}"))
        elif kind == "NOTE":
            out_ps.append(clone_text(T_NOTE, b[1]))
        prev = kind
    assert counter[0] == n_fn, f"각주 {n_fn}개 중 {counter[0]}개만 본문에서 찾음"
    for c in list(r1):
        r1.remove(c)
    for p in out_ps:
        r1.append(p)

    def para_with_text(root, startswith):
        return next(p for p in root if ptext(p).strip().startswith(startswith))
    set_single_text(para_with_text(r0, "공중전 인공지능 모델의"), d["title"] + " ")
    set_single_text(para_with_text(r0, "- 1대1 도그파이트"), d["subtitle"])
    set_single_text(para_with_text(r0, "(Explainability"), d["eng_title_paren"])
    set_single_text(para_with_text(r0, "본 연구는"), d["abstract_ko"])
    set_single_text(para_with_text(r0, "주제어"), d["keywords_ko"])

    ref_ps = [p for p in r2 if p.get("styleIDRef") == "17"]
    anchor = r2.index(ref_ps[0]); T_REF = ref_ps[0]
    for p in ref_ps:
        r2.remove(p)
    for j, ref in enumerate(d["refs"]):
        p = copy.deepcopy(T_REF); strip_layout(p)
        for r in p.findall(q("run")):
            p.remove(r)
        for m, part in enumerate(re.split(r"\*([^*]+)\*", ref)):
            if part:
                r = etree.SubElement(p, q("run")); r.set("charPrIDRef", "60" if m % 2 else "34")
                etree.SubElement(r, q("t")).text = clean(part)
        r2.insert(anchor + j, p)
    set_single_text(para_with_text(r2, "Explainability–Performance"), d["eng_title"])
    set_single_text(para_with_text(r2, "This study"), d["abstract_en"])
    set_single_text(para_with_text(r2, "Keywords"), d["keywords_en"])
    for t in r2.iter(q("t")):
        if t.text:
            t.text = (t.text.replace(OLD_TITLE_KO, d["title"].replace("<br>", " ")).replace(OLD_TITLE_EN, d["eng_title"])
                      .replace("관계을", "관계를"))

    def ser(root):
        return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    hpf = zin.read("Contents/content.hpf").decode("utf-8")
    for item_id, path in new_items:
        hpf = hpf.replace('<opf:item id="section1"', f'<opf:item id="{item_id}" href="BinData/{item_id}.png" '
                          f'media-type="image/png" isEmbeded="1"/><opf:item id="section1"', 1)
    last_bin = max(n for n in zin.namelist() if n.startswith("BinData/"))
    with zipfile.ZipFile(out, "w") as zout:
        for info in zin.infolist():
            name = info.filename
            data = {"Contents/section0.xml": lambda: ser(r0), "Contents/section1.xml": lambda: ser(r1),
                    "Contents/section2.xml": lambda: ser(r2),
                    "Contents/content.hpf": lambda: hpf.encode("utf-8")}.get(name, lambda: zin.read(name))()
            ct = zipfile.ZIP_STORED if name == "mimetype" else zipfile.ZIP_DEFLATED
            zout.writestr(zipfile.ZipInfo(name, date_time=info.date_time), data, compress_type=ct)
            if name == last_bin:
                for item_id, path in new_items:
                    zout.write(path, f"BinData/{item_id}.png", compress_type=zipfile.ZIP_DEFLATED)
    return dict(paragraphs=len(out_ps), footnotes=counter[0], tables=tbl_seq[0], new_pics=len(new_items),
                refs=len(d["refs"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pic", action="append", default=[], help='"<그림 n-n>=원본 그림 순번(0부터)"')
    ap.add_argument("--new-pic", action="append", default=[], help='"<그림 n-n>=경로.png:폭"')
    a = ap.parse_args()
    pic_map = {k: int(v) for k, v in (x.split("=", 1) for x in a.pic)}
    new_pics = {}
    for x in a.new_pic:
        k, v = x.split("=", 1); path, w = v.rsplit(":", 1); new_pics[k] = (path, int(w))
    print(build(a.md, a.template, a.out, pic_map, new_pics))


if __name__ == "__main__":
    main()
