"""Measure retrieval quality and log it, so every tuning change is comparable.

Usage:
  python eval.py             # retrieval only (free, fast): hit rate@k + MRR
  python eval.py --answers   # also generates answers and checks expected keywords (uses the LLM)

Each run appends one row to logs/eval_history.csv together with the settings used.
"""
import argparse
import csv
import json
import time
from pathlib import Path

from common import CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL, LLM_MODEL, TOP_K
from rag import answer, retrieve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="eval_set.json")
    ap.add_argument("--answers", action="store_true")
    args = ap.parse_args()

    items = json.loads(Path(args.file).read_text(encoding="utf-8"))
    hit_count, rr_sum, kw_scores = 0, 0.0, []

    for it in items:
        sources = [h["source"] for h in retrieve(it["question"])]
        exp = it["expected_source"]
        found = exp in sources
        if found:
            hit_count += 1
            rr_sum += 1 / (sources.index(exp) + 1)
        line = f"{'HIT ' if found else 'MISS'}  {it['question']}"
        if args.answers and it.get("expected_keywords"):
            text = answer(it["question"])[0].lower()
            cov = sum(k.lower() in text for k in it["expected_keywords"]) / len(it["expected_keywords"])
            kw_scores.append(cov)
            line += f"   (keywords {cov:.0%})"
        print(line)
    n = len(items)
    hit_rate, mrr = hit_count / n, rr_sum / n
    kw = sum(kw_scores) / len(kw_scores) if kw_scores else ""
    print(f"\nhit_rate@{TOP_K}: {hit_rate:.2f} | MRR: {mrr:.2f}" + (f" | keywords: {kw:.2f}" if kw != "" else ""))
    path = Path("logs/eval_history.csv")
    path.parent.mkdir(exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "n", "chunk_size", "overlap", "top_k", "embed_model", "llm", "hit_rate", "mrr", "keyword_cov"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), n, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, EMBED_MODEL, LLM_MODEL, round(hit_rate, 3), round(mrr, 3), kw if kw == "" else round(kw, 3)])
if __name__ == "__main__":
    main()
