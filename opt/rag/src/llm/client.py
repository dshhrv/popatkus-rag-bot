import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(ROOT))
import json
import pandas as pd
import requests
import re
import time
import requests
import os
from collections import defaultdict

from src.retrieval.bm25 import bm25_search, load_index, INDEX_PATH
from src.retrieval.dense import dense_search, COLLECTION_NAME
from src.retrieval.rrf import rrf_fuse
from src.retrieval.glossary import detect_terms, get_definitions
from scripts.translate import en2ru
from src.retrieval.retrieve import retrieve_top

REFUSAL_RU = "В документе нет прямого подтверждения"
REFUSAL_EN = "No direct confirmation"

from dotenv import load_dotenv
load_dotenv()

import os
api_key = os.getenv("OPENROUTER_API_KEY")


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


ONLY_ENGLISH = False
OUT_PATH_ALL = "data/popatkus_all_v5.jsonl"
IN_PATH = "data/golden_set.jsonl"

CLAUSE_RE = re.compile(r"^\s*(?P<id>\d+(?:\.\d+)*)\s*[\.\)]\s*(?P<body>.+\S)\s*$")
BRACKET_ID_RE = re.compile(r"\[([^\]\n]{1,120})\]")

# MODEL = "qwen2.5:1.5b-instruct"
MODEL = "openrouter"
URL = "http://localhost:11434/api/chat"
OUT_PATH = f"data/runs_llm_{MODEL}.jsonl"



PROMT = """
Ты - извлекатель цитат из контекста.

Тебе дан ВОПРОС и КОНТЕКСТ. Каждая строка КОНТЕКСТА начинается с [МЕТКА], где МЕТКА - это либо число (например [123]),
либо путь (например [Glossary, Академическая задолженность]).

ЗАДАЧА:
Верни ТОЛЬКО те строки из КОНТЕКСТА, которые прямо отвечают на вопрос.

СТРОГИЕ ПРАВИЛА (обязательно):
1) Нельзя добавлять НИ ОДНОГО слова, которого нет в выбранных строках контекста.
2) Нельзя перефразировать. Цитаты копируются дословно.
3) Каждая строка ответа ДОЛЖНА начинаться с [МЕТКА] ровно как в контексте, включая запятые и пробелы.
4) Выбери от 1 до 3 строк. Если одной строки достаточно - верни одну.
5) Если в контексте нет прямого ответа - выведи ровно одну строку:
В документе нет прямого подтверждения
6) Никаких списков, нумерации, заголовков, объяснений, комментариев.
7) Запрещено выводить [ID]. Если не знаешь метку - выведи "В документе нет прямого подтверждения".

ПРОВЕРКА ПЕРЕД ВЫВОДОМ:
- Убедись, что каждая строка ответа встречается в КОНТЕКСТЕ дословно.
- Убедись, что [МЕТКА] совпадает символ-в-символ.
"""


bm25, ids, meta = load_index(INDEX_PATH)


def citation_id(chunk):
    cl = chunk.get("clause_id")
    if cl is not None:
        cl = str(cl).strip()
        if cl:
            return cl
    hp = chunk.get("heading_path")
    hp = ", ".join(str(x).strip() for x in hp if str(x).strip())
    return hp



def load_chunks_map(jsonl_path):
    m = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            m[o["id"]] = o
    return m


chunks_map = load_chunks_map(OUT_PATH_ALL)
ALL_CLAUSE_IDS = set()
for ch in chunks_map.values():
    cid = citation_id(ch)
    if cid:
        ALL_CLAUSE_IDS.add(cid)


def extract_clause_ids(answer):
    out = set(BRACKET_ID_RE.findall(answer))
    return {x for x in out if x in ALL_CLAUSE_IDS}


def dump_line(f, obj: dict):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def clause_to_id(chunks_map):
    d = defaultdict(set)
    for chunk_id, ch in chunks_map.items():
        cid = citation_id(ch)
        if cid:
            d[cid].add(chunk_id)
    return d


def llm_metrics(answer, ctx_clause_ids, ctx_ids, rel, chunks_map, clause_to_chunk_ids):
    cited = extract_clause_ids(answer)
    cite_any = int(bool(cited))
    if cite_any:
        supported = [c for c in cited if c in ctx_clause_ids]
        cite_supported_rate = len(supported) / len(cited)
    else:
        cite_supported_rate = 0.0
    rel_set = set(rel or [])
    cited_chunk_ids = set()
    for cl in cited:
        cited_chunk_ids |= clause_to_chunk_ids.get(cl, set())
    cite_rel_any = int(bool(cited_chunk_ids & rel_set))
    hit_in_ctx = int(bool(set(ctx_ids) & rel_set))
    no_hit = int(not hit_in_ctx)
    abstain_ok = 0
    if no_hit:
        a_low = (answer or "").lower()
        abstain_ok = int((REFUSAL_RU in a_low) or (REFUSAL_EN in a_low))

    return {
        "cite_any": cite_any,
        "cite_supported_rate": cite_supported_rate,
        "cite_rel_any": cite_rel_any,
        "no_hit_in_ctx": no_hit,
        "abstain_ok": abstain_ok,
        "cited_clause_ids": sorted(cited),
    }




def initialize(in_path=IN_PATH, out_path=OUT_PATH, clause_to_chunk_ids=None, top_ctx=3):
    agg = {"n":0, "cite_any":0, "cite_rel_any":0, "sup_sum":0.0, "no_hit":0, "abstain_ok":0}
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "a", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            query = obj["text"]
            cid = obj["id"]
            rel = obj["rel"]
            lang = obj["lang"]
            final_ids, definitions = retrieve_top(query, lang,
                                                    bm25, ids, meta,
                                                    top_dense=80,
                                                    top_bm25 = 10,
                                                    top_final=10,
                                                    only_english=ONLY_ENGLISH)
            ctx_ids = final_ids[:top_ctx]
            clauses_text = "" 
            ctx_clause_ids = set()
            for chunk_id in ctx_ids:
                chunk = chunks_map.get(chunk_id)
                cid_txt = citation_id(chunk)
                ctx_clause_ids.add(cid_txt)
                clauses_text += f"[{cid_txt}] {chunk.get('text','')}\n"     
            user_content = f"ВОПРОС:\n{query}\nКОНТЕКСТ:\n{clauses_text}"
            messages = [{"role": "system", "content": PROMT},
                        {"role": "user", "content": user_content}]
            # flush = {
            #     "model": MODEL, 
            #     "messages": messages,
            #     "stream": False,
            #     "options": {
            #         "temperature": 0.0,
            #         "num_ctx": 1024,
            #         "num_predict": 256
            #     },
            #     "keep_alive": "5m"
            # }
            headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            }

            payload = {
                "model": "openai/gpt-4.1",
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 256,
            }

            resp = post_with_retries(session, OPENROUTER_URL, headers, payload, timeout=120)
            answer = resp.json()["choices"][0]["message"]["content"]
            # resp = requests.post(OPENROUTER_URL, json=flush, timeout=120)
            # resp.raise_for_status()
            # data = resp.json()
            # answer = data["message"]["content"]
            m = llm_metrics(
                answer=answer,
                ctx_clause_ids=ctx_clause_ids,
                ctx_ids=ctx_ids,
                rel=rel,
                chunks_map=chunks_map,
                clause_to_chunk_ids=clause_to_chunk_ids,
            )

            rec = {
                "id": cid,
                "lang": lang,
                "query": query,
                "rel": rel,
                "ctx_ids": ctx_ids,
                "answer": answer,
                "llm_metrics": m,
            }
            dump_line(fout, rec)
            agg["n"] += 1
            agg["cite_any"] += m["cite_any"]
            agg["cite_rel_any"] += m["cite_rel_any"]
            agg["sup_sum"] += m["cite_supported_rate"]
            
            if m["no_hit_in_ctx"]:
                agg["no_hit"] += 1
                agg["abstain_ok"] += m["abstain_ok"]
                
                
    n = max(1, agg["n"])
    print({
        "n": agg["n"],
        "cite_any_rate": agg["cite_any"]/n,
        "cite_rel_any_rate": agg["cite_rel_any"]/n,
        "cite_supported_rate_avg": agg["sup_sum"]/n,
        "abstain_ok_rate_when_no_hit": agg["abstain_ok"]/max(1, agg["no_hit"]),
    })



if __name__ == "__main__":
    import argparse
    # MODEL_TAG = MODEL.replace(":", "_")
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="data/golden_set.jsonl")
    parser.add_argument("--runs", default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-dir", default=None)
    
    args = parser.parse_args()
    if args.runs is None:
        args.runs = f"data/runs_llm/llm_LLAMA.jsonl"
    if args.out_csv is None:
        args.out_csv = f"csv/metrics_llm_LLAMA.csv"
    if args.out_dir is None:
        args.out_dir = f"graph/_llm_LLAMA"
    
        
    clause_to_chunk_ids = clause_to_id(chunks_map)
    initialize(in_path=args.golden, out_path=args.runs, clause_to_chunk_ids=clause_to_chunk_ids)
    df = pd.read_json(args.runs, lines=True)
    df = pd.json_normalize(df.to_dict(orient="records"), sep=".")
    df.columns = [c.replace("llm_metrics.", "") for c in df.columns]
    df.to_csv(args.out_csv, index=False)
