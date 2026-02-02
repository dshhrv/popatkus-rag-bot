from src.retrieval.bm25 import bm25_search
from src.retrieval.dense import dense_search, COLLECTION_NAME
from src.retrieval.rrf import rrf_fuse
from src.retrieval.glossary import detect_terms, get_definitions
from scripts.translate import en2ru


ONLY_ENGLISH = True

def retrieve_top30(query, lang, bm25, ids, meta, top_each=100, top_final=30, only_english=ONLY_ENGLISH):
    if only_english==True:
        query = en2ru(query)
        lang = "ru"
    terms = detect_terms(query, lang)
    definitions = []
    for t in terms:
        d = get_definitions(t, lang)
        if d:
            definitions.append(d)
    ngram_n = meta.get("ngram_n", 2)
    bm25_ids_ranked = bm25_search(bm25=bm25, ids=ids, query_text=query, lang=lang, top_k=top_each, ngram_n=ngram_n)
    dense_ids_ranked = dense_search(coll_name=COLLECTION_NAME, lang=lang, text=query, limit=top_each)
    final_ids = rrf_fuse(ranked_lists={"dense": dense_ids_ranked, "bm25": bm25_ids_ranked},
    weights={"dense": 1.5, "bm25": 1.0},
    k=60,
    key=lambda r: r,
    top=top_final)
    return final_ids, definitions
