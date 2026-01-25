import json
import docx
import re
from docx.table import Table, _Cell
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from hashlib import sha1
from docx.text.paragraph import Paragraph


DOCX_PATH = "popatkus_ru_ready.docx"
OUT_PATH = "popatkus_ru_v6.jsonl"
DOC_ID = "popatkus_ru"
LANG = "ru"
GLOSS_RE = re.compile(r"^\s*(?P<term>.{3,40}?)\s+[—–-]\s+(?P<def>.+\S)\s*$")
TERM_RE = re.compile(r"([A-ZА-ЯЁ])+.+")
CLAUSE_RE = re.compile(r"^\s*(?P<id>\d+(?:\.\d+)*)\s*[\.\)]\s*(?P<body>.+\S)\s*$")


def clean(s):
    return (s or "").replace("\u00A0", " ").strip()

def iter_children(parent):
    if hasattr(parent, 'element') and hasattr(parent.element, 'body'):
        parent_elm = parent.element.body
        parent_obj = parent
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
        parent_obj = parent
    elif hasattr(parent, "_element"):
        parent_elm = parent._element
        parent_obj = parent
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent_obj)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent_obj)
        else:
            continue


def table_to_text(tbl) :
    lines = []
    for row in tbl.rows:
        cells = []
        for cell in row.cells:
            cell_txt = " ".join(clean(p.text) for p in cell.paragraphs if clean(p.text))
            cells.append(clean(cell_txt))
        line = clean(" | ".join([c for c in cells if c]))
        if line:
            lines.append(line)
    return "\n".join(lines)

def make_id(*parts, n=24):
    def norm(x):
        if x is None:
            return ""
        if isinstance(x, (list, tuple)):
            return " > ".join(norm(i) for i in x if i not in (None, ""))
        return str(x)
    payload = "|".join(norm(p) for p in parts)
    return sha1(payload.encode("utf-8")).hexdigest()[:n]


def dump_line(f, obj: dict):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def export_jsonl(docx_path: str, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        doc = docx.Document(docx_path)
        heading1 = None
        current = None
        def start_clause(clause_id, cleaned_text, definition, term=None, heading=None, type=None):
            nonlocal current
            current = {
                "id": make_id(DOC_ID, LANG, type, heading, clause_id, term, cleaned_text),
                "clause_id": clause_id,
                "lang": LANG,
                "type": type,
                "term": term,
                "definition": [] if definition is None else [clean(definition)],
                "text_parts": [cleaned_text],
                "heading_path": heading,
                "order_start": None
            }
            return current


        def flush():
            nonlocal current
            if current is None:
                return
            text = "\n".join(current["text_parts"])
            definition = "\n".join(current["definition"])
            if current["type"] == "glossary":
                obj = {
                    "id": current["id"],
                    "doc_id": DOC_ID,
                    "lang": LANG,
                    "type": current["type"],
                    "text": text,
                    "meta": {
                        "source_file": DOCX_PATH,
                        "term": current["term"],
                        "definition": definition,
                        "version": OUT_PATH
                    }
                }
                dump_line(f, obj)
                current = None
            else:
                obj = {
                    "id": current["id"],
                    "doc_id": DOC_ID,
                    "clause_id": current["clause_id"],
                    "lang": LANG,
                    "type": current["type"],
                    "text": text,
                    "heading_path": current["heading_path"],
                    "meta": {
                        "source_file": DOCX_PATH,
                        "version": OUT_PATH
                    }
                }
                dump_line(f, obj)
                current = None

        for child in iter_children(doc):
            if isinstance(child, Paragraph):
                if child.style.name == "Heading 1":
                    flush()
                    heading1 = clean(child.text)
                elif child.style.name == "Normal":
                    m = GLOSS_RE.match(child.text)
                    if m and TERM_RE.match(m.group("term")):
                        flush()
                        start_clause(None, cleaned_text=clean(child.text), term=clean(m.group("term")), definition=clean(m.group("def")), type="glossary", heading=heading1)
                    elif CLAUSE_RE.match(child.text):
                        t = CLAUSE_RE.match(child.text)
                        flush()
                        heading_path = [heading1] if heading1 else []
                        start_clause(t.group("id"), clean(t.group("body")), term=None, definition=None, type="rules", heading=heading_path)
                    else:
                        if current is not None:
                            current["text_parts"].append(clean(child.text))
                            if current["type"] == "glossary":
                                current["definition"].append(clean(child.text))

            if isinstance(child, Table):
                tbl = table_to_text(child)
                if current is not None:
                    current["text_parts"].append(tbl)
        flush()


if __name__ == "__main__":
    export_jsonl(
        docx_path=DOCX_PATH,
        out_path=OUT_PATH,
    )

