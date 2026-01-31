import argparse
import sys
import json
import math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.retrieval.dense import dense_search, COLLECTION_NAME
from src.retrieval.bm25 import bm25_search, load_index, INDEX_PATH
from src.retrieval.rrf import rrf_fuse
def norm_id(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return x.get("sha1") or x.get("id") or str(x)
    if hasattr(x, "payload"):
        return x.payload.get("sha1") or str(getattr(x, "id", x))
    return str(x)

def dump_line(f, obj):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def mrr_at_k(ranked, rel_set, k):
    for i, d in enumerate(ranked[:k], start=1):
        if d in rel_set:
            return 1.0/i
    return 0.0


def ndcg_at_k(ranked, rel_set, k):
    m = min(k, len(rel_set))
    if m == 0:
        return 0.0
    dcg = 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if d in rel_set:
            dcg += 1.0 / math.log2(i + 1)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, m + 1))
    return dcg / idcg if idcg > 0 else 0.0


def plot(agg, out_png, metric, title):
    plt.figure()
    plt.plot(agg["k"], agg[metric], marker="o")
    plt.xlabel("K")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)


def run_retrieval(mode, golden_path, runs_path, bm25, ids, meta, top_each, top_final, weights, k0):
    runs_path = Path(runs_path)
    runs_path.parent.mkdir(parents=True, exist_ok=True)

    with open(golden_path, "r", encoding="utf-8") as fin, open(runs_path, "w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            qid = obj["id"]
            lang = obj.get("lang", "ru")
            text = obj["text"]

            if mode == "dense":
                ranked = dense_search(coll_name=COLLECTION_NAME, lang=lang, text=text, limit=top_each)
            elif mode == "bm25":
                ngram_n = meta.get("ngram_n", 2)
                ranked = bm25_search(bm25=bm25, ids=ids, query_text=text, lang=lang, top_k=top_each, ngram_n=ngram_n)
            elif mode == "rrf":
                ngram_n = meta.get("ngram_n", 2)
                bm25_ranked = bm25_search(bm25=bm25, ids=ids, query_text=text, lang=lang, top_k=top_each, ngram_n=ngram_n)
                dense_ranked = dense_search(coll_name=COLLECTION_NAME, lang=lang, text=text, limit=top_each)
                ranked = rrf_fuse(
                    ranked_lists={"dense": dense_ranked, "bm25": bm25_ranked},
                    weights=weights,
                    k=k0,
                    key=lambda r: r,
                    top=top_final,
                )
            ranked = [norm_id(x) for x in ranked]
            dump_line(fout, {"id": qid, "retrieved": ranked})


def eval_metrics(golden_path, runs_path, ks):
    rel_by_id = {}
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rel_by_id[obj["id"]] = set(obj.get("rel", []))
    rows = []
    with open(runs_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = obj["id"]
            ranked = obj["retrieved"]
            rel = rel_by_id.get(qid, set())
            for k in ks:
                topk = ranked[:k]
                hit_cnt = sum(1 for d in topk if d in rel)
                rows.append(
                    {
                        "id": qid,
                        "k": k,
                        "recall": hit_cnt / len(rel) if rel else 0.0,
                        "precision": hit_cnt / k if k else 0.0,
                        "hit": 1.0 if hit_cnt > 0 else 0.0,
                        "mrr": mrr_at_k(ranked, rel, k),
                        "ndcg": ndcg_at_k(ranked, rel, k),
                    }
                )
    df = pd.DataFrame(rows)
    agg = df.groupby("k", as_index=False).mean(numeric_only=True)
    return df, agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="data/golden_set.jsonl")
    parser.add_argument("--mode", choices=["dense", "bm25", "rrf"], required=True)
    parser.add_argument("--top-each", type=int, default=100)
    parser.add_argument("--top-final", type=int, default=30)
    parser.add_argument("--rrf-k0", type=int, default=60)
    parser.add_argument("--dense-w", type=float, default=1.0)
    parser.add_argument("--bm25-w", type=float, default=1.0)
    parser.add_argument("--ks", default="1,3,5,10,20,30")

    parser.add_argument("--runs", default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    if args.runs is None:
        args.runs = f"data/runs_{args.mode}.jsonl"
    if args.out_csv is None:
        args.out_csv = f"csv/metrics_{args.mode}.csv"
    if args.out_dir is None:
        args.out_dir = f"graph/{args.mode}"
    ks = [int(x) for x in args.ks.split(",")]
    bm25, ids, meta = load_index(INDEX_PATH)
    weights = {"dense": args.dense_w, "bm25": args.bm25_w}

    run_retrieval(
        mode=args.mode,
        golden_path=args.golden,
        runs_path=args.runs,
        bm25=bm25,
        ids=ids,
        meta=meta,
        top_each=args.top_each,
        top_final=args.top_final,
        weights=weights,
        k0=args.rrf_k0,
    )
    _, agg = eval_metrics(args.golden, args.runs, ks)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.out_csv, index=False)
    plot(agg, str(out_dir / f"{args.mode}_ndcg.png"), "ndcg", f"{args.mode}: nDCG@K")
    plot(agg, str(out_dir / f"{args.mode}_recall.png"), "recall", f"{args.mode}: Recall@K")
    plot(agg, str(out_dir / f"{args.mode}_mrr.png"), "mrr", f"{args.mode}: MRR@K")
    plot(agg, str(out_dir / f"{args.mode}_precision.png"), "precision", f"{args.mode}: Precision@K")


if __name__ == "__main__":
    main()
