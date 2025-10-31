# build_faiss_per_file_nochunk.py
"""
Create a FAISS vectorstore per JSON file using HuggingFace embeddings.
Assumptions:
- Each JSON file is a list of dicts, each dict is already a chunk, e.g.
  [{"header": "...", "text": "...", "page": N}, ...]
- We will not further split chunks.
- Uses a HuggingFace sentence-transformers model by default.
"""

from pathlib import Path
import json
from typing import List, Iterable

# ---- Robust imports to support LangChain v0.3+ and older versions ----
# Document
from langchain_core.documents import Document  # new
from langchain_community.vectorstores import FAISS  # new
from langchain_community.embeddings import HuggingFaceEmbeddings  # new

# ---------------------------------------------------------------------

# ---------- CONFIG ----------
DATA_DIR = Path(r"C:\Users\aadar\Documents\pdf\output")        # folder with your chunked JSON files
OUTPUT_DIR = Path(r"C:\Users\aadar\Documents\pdf\FaissIndex")   # parent folder to store indexes
BUILD_GLOBAL_INDEX = True            # set False to skip building a global index
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # change if desired
# ----------------------------

def iter_documents_from_json(path: Path) -> Iterable[Document]:
    """
    Yield Documents from a single JSON file.
    Each JSON element is treated as a pre-chunked document/chunk.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Skipping {path.name} (failed to read/parse): {e}")
        return

    if not isinstance(data, list):
        print(f"[WARN] {path.name} is not a list — skipping.")
        return

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        header = (item.get("header") or "").strip()
        text = (item.get("text") or "").strip()
        page = item.get("page")
        # If JSON chunk already contains full chunk text, prefer text; include header if present
        if header and text:
            content = f"{header}\n\n{text}"
        else:
            content = header or text

        if not content.strip():
            # skip empty chunks
            continue

        metadata = {
            "source_file": path.name,
            "page": page,
            "header": header,
            "chunk_index": idx
        }
        yield Document(page_content=content, metadata=metadata)

def get_embeddings():
    """
    Instantiate HuggingFaceEmbeddings.
    This will download the model the first time (sentence-transformers).
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def build_index_for_file(json_path: Path, embeddings):
    docs = list(iter_documents_from_json(json_path))
    if not docs:
        print(f"[INFO] No docs/chunks in {json_path.name} — skipping.")
        return None, []

    print(f"[{json_path.name}] indexing {len(docs)} chunks...")

    # Build FAISS store directly from documents (no further splitting)
    store = FAISS.from_documents(docs, embeddings)

    out_dir = OUTPUT_DIR / json_path.stem / "faiss_index"
    out_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(out_dir))
    print(f"[OK] Saved index for {json_path.name} -> {out_dir}")
    return store, docs

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(DATA_DIR.glob("*.json"))
    if not json_files:
        print(f"[ERROR] No JSON files found in {DATA_DIR.resolve()}. Place your JSONs there.")
        return

    print(f"[INFO] Using HuggingFace model: {EMBEDDING_MODEL}")
    embeddings = get_embeddings()

    all_docs: List[Document] = []
    for p in json_files:
        print("\n---")
        print(f"Processing: {p.name}")
        store, docs = build_index_for_file(p, embeddings)
        all_docs.extend(docs)

    if BUILD_GLOBAL_INDEX and all_docs:
        print(f"\n[INFO] Building GLOBAL index from {len(all_docs)} chunks...")
        global_store = FAISS.from_documents(all_docs, embeddings)
        global_dir = OUTPUT_DIR / "_GLOBAL" / "faiss_index"
        global_dir.mkdir(parents=True, exist_ok=True)
        global_store.save_local(str(global_dir))
        print(f"[OK] Saved GLOBAL index -> {global_dir}")

    print("\nAll done.")

if __name__ == "__main__":
    main()
