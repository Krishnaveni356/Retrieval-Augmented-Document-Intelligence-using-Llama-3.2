"""Retrieve -> build prompt -> generate.

Usage:
  python rag.py                       # interactive chat
  python rag.py "your question"       # one-shot
Every query is logged to logs/queries.jsonl (latency, top score, sources) for tracking.
"""
import json
import sys
import time
from pathlib import Path
from common import LLM_MODEL, LLM_PROVIDER, MIN_SCORE, OLLAMA_URL, TOP_K, embed, get_collection
SYSTEM = (
       "Answer the question using the provided context. You may combine and summarize "
       "information from several passages. Cite sources inline like [1], [2], matching the context numbers. "
       "Say you don't know only if the context has nothing relevant to the question. "
       "Do not use outside knowledge."
         )
NOT_FOUND = "I couldn't find anything relevant in the documents."
LOG_FILE = Path("logs/queries.jsonl")
def retrieve(query, k=TOP_K):
    res = get_collection().query(query_embeddings=embed([query]), n_results=k)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append(
            {"text": doc, "source": meta["source"], "page": meta.get("page") or None, "score": 1 - dist}
        )
    return hits  # already sorted best-first
def build_prompt(question, hits):
    blocks = []
    for i, h in enumerate(hits, 1):
        loc = f"{h['source']}, p.{h['page']}" if h["page"] else h["source"]
        blocks.append(f"[{i}] ({loc})\n{h['text']}")
    return "Context:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
def generate(prompt):
    if LLM_PROVIDER == "ollama":
        import requests
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": LLM_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=180,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]
    import anthropic  # reads ANTHROPIC_API_KEY from the environment
    msg = anthropic.Anthropic().messages.create(
        model=LLM_MODEL,
        max_tokens=800,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")
def answer(question, k=TOP_K):
    t0 = time.time()
    hits = retrieve(question, k)
    answered = bool(hits) and hits[0]["score"] >= MIN_SCORE
    text = generate(build_prompt(question, hits)) if answered else NOT_FOUND
    LOG_FILE.parent.mkdir(exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "question": question,
                    "answered": answered,
                    "top_score": round(hits[0]["score"], 3) if hits else None,
                    "sources": [h["source"] for h in hits],
                    "latency_ms": int((time.time() - t0) * 1000),
                }
            )
            + "\n"
        )
    return text, hits
def show(question):
    text, hits = answer(question)
    print("\n" + text + "\n\nSources:")
    for i, h in enumerate(hits, 1):
        page = f", p.{h['page']}" if h["page"] else ""
        print(f"  [{i}] {h['source']}{page}  (score {h['score']:.2f})")
if __name__ == "__main__":
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
    else:
        print("RAG ready. Ask a question (Ctrl+C to quit).")
        while True:
            try:
                q = input("\n> ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if q:
                show(q)
