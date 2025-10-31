# merge_header_tuples_with_folder.py
# pip install "pymupdf>=1.21,<2"

import json
from pathlib import Path
import fitz  # PyMuPDF

# ------------ CONFIGURATION ------------
PDF_PATH = r"C:\Users\aadar\OneDrive\Desktop\PROJECT\Department of Political Science.pdf"
OUTPUT_FOLDER = r"C:\Users\aadar\OneDrive\Desktop\pdf\output" # <--- change this folder path
# ---------------------------------------


def infer_style(fontname: str) -> str:
    name = (fontname or "").lower()
    is_bold = any(k in name for k in ["bold", "black", "heavy", "semibold", "demi"])
    is_italic = any(k in name for k in ["italic", "oblique"])
    is_mono = any(k in name for k in ["mono", "courier", "code", "console"])
    parts = []
    if is_bold: parts.append("bold")
    if is_italic: parts.append("italic")
    if is_mono: parts.append("monospace")
    return ", ".join(parts) if parts else "regular"


def is_header_span(span) -> bool:
    font = span.get("font", "")
    size = float(span.get("size", 0) or 0)
    style = infer_style(font)
    print(font)
    print(size)
    return (
        font == "TimesNewRomanPS-BoldMT"
        and size >= 12
        #and "bold" in style
    )


def extract_header_tuples(pdf_path: str):
    tuples = []
    with fitz.open(pdf_path) as doc:
        collecting = False
        current_header, current_text_chunks, current_page_for_header = None, [], None

        for page_num, page in enumerate(doc, start=1):
            info = page.get_text("dict")
            for block in info.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    i = 0
                    while i < len(spans):
                        sp = spans[i]
                        text = sp.get("text", "") or ""

                        if is_header_span(sp):
                            if collecting:
                                tuples.append(
                                    (current_header.strip(),
                                     "".join(current_text_chunks).strip(),
                                     current_page_for_header)
                                )
                                current_header, current_text_chunks, collecting, current_page_for_header = None, [], False, None

                            header_parts = [text]
                            i += 1
                            while i < len(spans) and is_header_span(spans[i]):
                                header_parts.append(spans[i].get("text", "") or "")
                                i += 1

                            current_header = "".join(header_parts)
                            current_page_for_header = page_num
                            collecting = True
                            continue
                        else:
                            if collecting:
                                current_text_chunks.append(text)
                            i += 1
                    if collecting:
                        current_text_chunks.append("\n")

        if collecting and current_header is not None:
            tuples.append(
                (current_header.strip(),
                 "".join(current_text_chunks).strip(),
                 current_page_for_header)
            )
    return tuples


def merge_empty_headers(tuples):
    result, pending_headers = [], []
    pending_first_page, pending_last_page = None, None

    def flush_pending_as_individuals():
        nonlocal pending_headers, pending_first_page, pending_last_page
        for h in pending_headers:
            result.append((h.strip(), "", pending_first_page))
        pending_headers, pending_first_page, pending_last_page = [], None, None

    i, n = 0, len(tuples)
    while i < n:
        h, t, p = tuples[i]
        if not t.strip():
            if not pending_headers:
                pending_headers, pending_first_page, pending_last_page = [h], p, p
            else:
                if p == pending_last_page or p == pending_last_page + 1:
                    pending_headers.append(h)
                    pending_last_page = p
                else:
                    flush_pending_as_individuals()
                    pending_headers, pending_first_page, pending_last_page = [h], p, p
            i += 1
            continue

        if pending_headers:
            if p == pending_last_page or p == pending_last_page + 1:
                merged_header = " ".join([*pending_headers, h]).strip()
                result.append((merged_header, t, pending_first_page))
                pending_headers, pending_first_page, pending_last_page = [], None, None
            else:
                flush_pending_as_individuals()
                result.append((h, t, p))
        else:
            result.append((h, t, p))
        i += 1

    if pending_headers:
        merged_header = " ".join(pending_headers).strip()
        result.append((merged_header, "", pending_first_page))

    return result


if __name__ == "__main__":
    tuples = extract_header_tuples(PDF_PATH)
    merged = merge_empty_headers(tuples)

    # --- Save result in OUTPUT_FOLDER ---
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    out_file = Path(OUTPUT_FOLDER) / (Path(PDF_PATH).stem + ".merged_header_tuples.json")

    out = [{"header": h, "text": t, "page": p} for (h, t, p) in merged]
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(merged)} merged tuples to {out_file}")
