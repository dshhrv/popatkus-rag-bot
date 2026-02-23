import json
import argparse
import numpy as np
from sentence_transformers import CrossEncoder
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.retrieval.bm25 import load_index, INDEX_PATH
from src.retrieval.retrieve import retrieve_top
from src.retrieval.encoder import rerank_one, MODEL_RERANK


def retrieve_and_rerank(query, lang, bm25, ids, meta, reranker, chunks_map,
                        top_dense, top_bm25, top_final, batch_size):
    final_ids, definitions = retrieve_top(
        query=query, lang=lang, bm25=bm25, ids=ids, meta=meta,
        top_dense=top_dense, top_bm25=top_bm25, top_final=top_final
    )
    reranked = rerank_one(
        reranker=reranker, query=query, final_ids=final_ids,
        chunks_map=chunks_map, batch_size=batch_size
    )
    return reranked, definitions


def dump_line(f, obj):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    
def crag(reranked_items, query, chunks_map):
    scores = np.array([x["ce_score"] for x in reranked_items], dtype=float)
    s1 = float(scores[0])
    sn = float(scores.mean())
    gap = float(scores[0] - scores[len(scores)-1])
    std = float(scores.std())
    
    
    
    
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    
