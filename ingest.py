"""Load documents -> chunk -> embed -> store in ChromaDB.
Usage:  python ingest.py
Put .txt / .md / .pdf files in docs/ first. Re-running replaces old chunks for each file.
"""
import re
from common import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR, embed, get_collection
SUPPORTED = {".txt", ".md", ".pdf"}
def load_file(path):
    """Return a list of (text, page_number). page_number is 0 when not applicable."""
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return [(path.read_text(encoding="utf-8", errors="ignore"), 0)]
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return [(page.extract_text() or "", i + 1) for i, page in enumerate(reader.pages)]
    return []
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Sliding window that prefers to cut at paragraph / line / sentence / word boundaries."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window_start = start + int(size * 0.8)  # only snap inside the last 20% of the window
            for sep in ("\n\n", "\n", ". ", " "):
                cut = text.rfind(sep, window_start, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks
def main():
    files = [p for p in sorted(DOCS_DIR.rglob("*")) if p.suffix.lower() in SUPPORTED]
    if not files:
        raise SystemExit(f"No .txt/.md/.pdf files found in {DOCS_DIR}/ - add some and re-run.")
    col = get_collection()
    total = 0
    for path in files:
        source = str(path.relative_to(DOCS_DIR))
        col.delete(where={"source": source})  # re-ingest = replace, no duplicates
        ids, docs, metas = [], [], []
        for text, page in load_file(path):
            for i, chunk in enumerate(chunk_text(text)):
                ids.append(f"{source}::p{page}::c{i}")
                docs.append(chunk)
                metas.append({"source": source, "page": page})
        for s in range(0, len(docs), 64):
            batch = docs[s : s + 64]
            col.upsert(
                ids=ids[s : s + 64],
                documents=batch,
                embeddings=embed(batch),
                metadatas=metas[s : s + 64],
            )
        total += len(docs)
        print(f"  {source}: {len(docs)} chunks")
    print(f"Done. {total} chunks from {len(files)} file(s) -> vector store.")
if __name__ == "__main__":
    main()
