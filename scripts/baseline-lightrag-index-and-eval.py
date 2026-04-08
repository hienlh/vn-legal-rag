#!/usr/bin/env python3
"""
LightRAG baseline: offline indexing + online evaluation on hard-200 benchmark.

Phase 1 (Offline): Insert all 840 legal articles into LightRAG with doc-qualified IDs.
Phase 2 (Online): Query each benchmark question, extract ranked article IDs, compute metrics.

LightRAG builds a knowledge graph from extracted entities/relations and uses vector+graph
retrieval. We test multiple query modes: naive (vector-only), hybrid (local+global KG).

Usage:
    # Full pipeline (index if needed + evaluate)
    python scripts/baseline-lightrag-index-and-eval.py -o results/baseline_lightrag.json

    # Re-index from scratch
    python scripts/baseline-lightrag-index-and-eval.py --reindex -o results/baseline_lightrag.json

    # Evaluate only (skip indexing, index must exist)
    python scripts/baseline-lightrag-index-and-eval.py --eval-only -o results/baseline_lightrag.json

    # Limit questions
    python scripts/baseline-lightrag-index-and-eval.py --limit 10 -o results/baseline_lightrag_10.json

    # Choose query mode (default: hybrid)
    python scripts/baseline-lightrag-index-and-eval.py --mode naive -o results/baseline_lightrag_naive.json

Dependencies:
    pip install lightrag[offline-llm]
"""

import argparse
import asyncio
import csv
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(override=True)


# ---------------------------------------------------------------------------
# Metrics (same as evaluate-full-rag-with-baselines.py)
# ---------------------------------------------------------------------------

K_VALUES = [1, 3, 5, 10, 20, 30]


def extract_expected_article_ids(article_ids: str) -> set:
    """Parse expected article IDs from benchmark CSV."""
    if not article_ids or article_ids.strip() == "":
        return set()
    articles = set()
    for ref in article_ids.split(";"):
        ref = ref.strip()
        if not ref:
            continue
        match = re.match(r'(.+?:d\d+)', ref)
        if match:
            articles.add(match.group(1))
    return articles


def calc_ir_metrics(expected: set, ranked: list) -> dict:
    """Calculate IR metrics for a single query."""
    if not expected or not ranked:
        return {"rr": 0.0, **{f"hit@{k}": 0 for k in K_VALUES},
                **{f"recall@{k}": 0.0 for k in K_VALUES}}

    rr = 0.0
    for i, a in enumerate(ranked):
        if a in expected:
            rr = 1.0 / (i + 1)
            break

    metrics = {"rr": rr}
    for k in K_VALUES:
        top_k = set(ranked[:k])
        relevant = len(expected & top_k)
        metrics[f"hit@{k}"] = 1 if relevant > 0 else 0
        metrics[f"recall@{k}"] = relevant / len(expected)
    return metrics


# ---------------------------------------------------------------------------
# Article loading from SQLite
# ---------------------------------------------------------------------------

def load_articles_from_db(db_path: str) -> list:
    """Load all articles from legal_docs.db with doc-qualified IDs.

    Returns list of dicts: {article_id, title, content, document_id}
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, article_number, title, content, document_id
        FROM legal_articles
        ORDER BY document_id, article_number
    """)
    articles = []
    for row in cur.fetchall():
        article_id, art_num, title, content, doc_id = row
        # Build readable text for indexing
        art_title = title or f"Điều {art_num}"
        text = f"Điều {art_num}. {art_title}\n\n{content or ''}"
        articles.append({
            "article_id": article_id,  # e.g. "59-2020-QH14:d206"
            "title": art_title,
            "content": text,
            "document_id": doc_id,
        })
    conn.close()
    return articles


# ---------------------------------------------------------------------------
# LightRAG setup
# ---------------------------------------------------------------------------

async def create_lightrag_instance(working_dir: str, llm_model: str, api_key: str, base_url: str = None):
    """Create and initialize a LightRAG instance."""
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    import anthropic

    # LLM function using httpx (proxy returns SSE stream, parse manually)
    import httpx

    async def llm_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs):
        messages = []
        if history_messages:
            for msg in history_messages:
                role = msg.get("role", "user")
                if role == "system":
                    continue
                messages.append({"role": role, "content": msg.get("content", "")})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": llm_model,
            "messages": messages,
            "max_tokens": 4096,
            "stream": True,
        }
        if system_prompt:
            body["system"] = system_prompt

        url = f"{base_url}/v1/messages" if base_url else "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    response = await client.post(url, json=body, headers=headers)
                    # Parse SSE stream to extract text
                    text_parts = []
                    for line in response.text.split("\n"):
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if data.get("type") == "content_block_delta":
                                    delta = data.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        text_parts.append(delta.get("text", ""))
                            except json.JSONDecodeError:
                                continue
                    return "".join(text_parts)
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise

    # Embedding function using sentence-transformers (local, free, multilingual)
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embed_dim = embed_model.get_sentence_embedding_dimension()

    async def embedding_func(texts: list[str]) -> "np.ndarray":
        import numpy as np
        embeddings = embed_model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings)

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embed_dim,
            max_token_size=512,
            func=embedding_func,
        ),
        # Use lightweight local storage
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
        doc_status_storage="JsonDocStatusStorage",
        # Chunking: each article is already a coherent unit, use large chunks
        chunk_token_size=2000,
        chunk_overlap_token_size=200,
        # 5 parallel workers — proxy handles this well (tested)
        max_parallel_insert=5,
        llm_model_max_async=5,
        entity_extract_max_gleaning=0,  # Skip gleaning to reduce LLM calls
        # Vietnamese legal docs — extract entities/relations in Vietnamese
        addon_params={"language": "Vietnamese"},
    )

    await rag.initialize_storages()
    # Required by newer LightRAG versions for pipeline to work
    try:
        from lightrag.kg.shared_storage import initialize_pipeline_status
        await initialize_pipeline_status()
    except ImportError:
        pass  # Older LightRAG version without pipeline_status
    return rag


# ---------------------------------------------------------------------------
# Phase 1: Offline Indexing
# ---------------------------------------------------------------------------

async def run_indexing(rag, articles: list, batch_size: int = 20):
    """Insert all articles into LightRAG in batches."""
    total = len(articles)
    print(f"\n  Indexing {total} articles into LightRAG...")

    for start in range(0, total, batch_size):
        batch = articles[start:start + batch_size]
        texts = [a["content"] for a in batch]
        ids = [a["article_id"] for a in batch]

        try:
            await rag.ainsert(texts, ids=ids)
        except Exception as e:
            print(f"  [WARN] Batch {start}-{start+len(batch)}: {e}")
            # Try one by one on failure
            for i, (text, aid) in enumerate(zip(texts, ids)):
                try:
                    await rag.ainsert([text], ids=[aid])
                except Exception as e2:
                    print(f"  [ERR] Article {aid}: {e2}")

        done = min(start + batch_size, total)
        print(f"  [{done}/{total}] articles indexed")

    print(f"  Indexing complete: {total} articles")


# ---------------------------------------------------------------------------
# Phase 2: Online Evaluation
# ---------------------------------------------------------------------------

def build_content_to_article_index(articles: list) -> dict:
    """Build index mapping article content prefix → article_id.

    Each article was inserted as "Điều {num}. {title}\n\n{content}".
    LightRAG chunks may contain the beginning of this text.
    We index by first 80 chars of content for fast lookup.
    """
    index = {}
    for a in articles:
        content = a["content"].strip()
        # Index by multiple prefix lengths for robustness
        for prefix_len in [80, 50, 30]:
            if len(content) >= prefix_len:
                key = content[:prefix_len]
                index[key] = a["article_id"]
    return index


def load_chunk_to_article_map(working_dir: str) -> dict:
    """Load chunk_id → article_id mapping from LightRAG's text_chunks store.

    Each chunk has a `full_doc_id` field that directly maps to the article_id
    (e.g., "59-2020-QH14:d206"). This is far more reliable than content prefix matching.
    """
    kv_path = os.path.join(working_dir, "kv_store_text_chunks.json")
    if not os.path.exists(kv_path):
        return {}
    with open(kv_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return {cid: cdata["full_doc_id"] for cid, cdata in chunks.items()
            if cdata.get("full_doc_id")}


def extract_article_ids_from_chunks(chunks: list, chunk_to_article: dict,
                                     content_index: dict, all_article_ids: set) -> list:
    """Extract ranked article IDs from LightRAG chunk results.

    Primary strategy: use chunk_id → full_doc_id direct mapping.
    Fallback: content prefix matching (for chunks without chunk_id).
    """
    ranked = []
    seen = set()

    for chunk in chunks:
        if isinstance(chunk, str):
            continue
        if not isinstance(chunk, dict):
            continue

        article_id = None

        # Strategy 1 (primary): direct chunk_id → article_id via full_doc_id
        chunk_id = chunk.get("chunk_id") or chunk.get("_id") or chunk.get("id")
        if chunk_id and chunk_id in chunk_to_article:
            article_id = chunk_to_article[chunk_id]

        # Strategy 2 (fallback): content prefix matching
        if not article_id:
            content = chunk.get("content", "").strip()
            if content:
                for prefix_len in [80, 50, 30]:
                    key = content[:prefix_len]
                    if key in content_index:
                        article_id = content_index[key]
                        break

        if article_id and article_id not in seen:
            ranked.append(article_id)
            seen.add(article_id)

    return ranked


async def evaluate_single_query(rag, question: str, mode: str,
                                chunk_to_article: dict, content_index: dict,
                                all_article_ids: set, top_k: int = 30):
    """Run a single query and return ranked article IDs."""
    from lightrag import QueryParam

    param = QueryParam(
        mode=mode,
        top_k=top_k,
        chunk_top_k=top_k,
        stream=False,
    )

    ranked = []
    try:
        data = await rag.aquery_data(question, param=param)

        # Extract from chunks inside data dict
        chunks = []
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], dict):
                chunks = data["data"].get("chunks", [])
            else:
                chunks = data.get("chunks", [])

        if chunks:
            ranked = extract_article_ids_from_chunks(
                chunks, chunk_to_article, content_index, all_article_ids)

    except Exception as e:
        return [], str(e)

    return ranked, None


async def run_evaluation(rag, benchmark_path: str, mode: str,
                         chunk_to_article: dict, content_index: dict,
                         all_article_ids: set, output_path: str,
                         start: int = 1, limit: int = None, workers: int = 1):
    """Run evaluation on benchmark dataset with optional parallel workers."""

    with open(benchmark_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # Apply start/limit
    tasks = []
    for i, row in enumerate(rows):
        if (i + 1) < start:
            continue
        if limit and len(tasks) >= limit:
            break
        tasks.append(row)

    print(f"\n  Evaluating {len(tasks)} questions (mode={mode}, workers={workers})...")
    print(f"  {'#':>4} {'STT':>5} {'Result':>6}  Running Hit@10")
    print("-" * 55)

    # Pre-build records and identify evaluable tasks
    records = []
    eval_indices = []  # indices into records that need evaluation
    for idx, row in enumerate(tasks):
        stt = row.get("STT", "?")
        question = row.get("Content", "") or row.get("question", "")
        article_ids = row.get("Article_IDs", "") or row.get("article_ids", "")
        category = row.get("Category", "")
        expected = extract_expected_article_ids(article_ids)
        record = {
            "stt": stt, "category": category, "question": question,
            "expected_articles": sorted(expected), "num_expected": len(expected),
        }
        if not expected:
            record["skipped"] = True
            records.append(record)
            print(f"[{idx+1:4d}] STT {stt:>5} SKIPPED")
        else:
            record["skipped"] = False
            records.append(record)
            eval_indices.append(idx)

    # Evaluate queries (parallel with semaphore)
    sem = asyncio.Semaphore(workers)
    completed = [0]  # mutable counter
    hits_10 = [0]
    tested = [0]
    lock = asyncio.Lock()

    async def eval_one(idx):
        record = records[idx]
        question = record["question"]
        expected = set(record["expected_articles"])

        async with sem:
            ranked, error = await evaluate_single_query(
                rag, question, mode, chunk_to_article, content_index, all_article_ids)

        if error:
            record["lightrag_error"] = error
            record["lightrag_retrieved"] = []
            for k in K_VALUES:
                record[f"lightrag_hit@{k}"] = 0
                record[f"lightrag_recall@{k}"] = 0.0
            record["lightrag_mrr"] = 0.0
        else:
            ir = calc_ir_metrics(expected, ranked)
            record["lightrag_retrieved"] = ranked[:30]
            for k in K_VALUES:
                record[f"lightrag_hit@{k}"] = ir[f"hit@{k}"]
                record[f"lightrag_recall@{k}"] = round(ir[f"recall@{k}"], 4)
            record["lightrag_mrr"] = round(ir["rr"], 4)

        async with lock:
            completed[0] += 1
            tested[0] += 1
            hit10 = record.get("lightrag_hit@10", 0)
            hits_10[0] += hit10
            rate = hits_10[0] / tested[0] * 100
            result_str = "HIT" if hit10 else "MISS"
            print(f"[{completed[0]:4d}] STT {record['stt']:>5} {result_str:>6}  {rate:.1f}%")
            # Incremental save
            _save_results(output_path, records, benchmark_path, mode,
                          tested[0], hits_10[0], "in_progress")

    await asyncio.gather(*[eval_one(i) for i in eval_indices])

    return records, tested[0], hits_10[0]


def _save_results(output_path: str, results: list, benchmark_path: str,
                  mode: str, tested: int, hits_10: int, status: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    non_skipped = [r for r in results if not r.get("skipped")]
    n = len(non_skipped)

    summary = {}
    if n > 0:
        summary = {
            "hit@10": f"{hits_10}/{tested} ({hits_10/tested*100:.1f}%)" if tested else "N/A",
            "mrr": round(sum(r.get("lightrag_mrr", 0) for r in non_skipped) / n, 4),
        }
        for k in K_VALUES:
            avg_hit = sum(r.get(f"lightrag_hit@{k}", 0) for r in non_skipped) / n
            summary[f"hit@{k}_rate"] = round(avg_hit * 100, 1)

    data = {
        "metadata": {
            "baseline": "LightRAG",
            "mode": mode,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_file": benchmark_path,
            "total_questions": len(results),
            "total_tested": tested,
            "status": status,
        },
        "summary": summary,
        "results": sorted(results, key=lambda r: int(r.get("stt", 0))),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main(args):
    db_path = args.db_path
    working_dir = args.working_dir
    benchmark_path = args.test_file
    output_path = args.output if args.output.endswith(".json") else f"{args.output}.json"

    # Load articles
    articles = load_articles_from_db(db_path)
    all_article_ids = {a["article_id"] for a in articles}
    print(f"  Loaded {len(articles)} articles from {db_path}")

    # Check for API key (Anthropic proxy) — prefer dotenv to avoid empty env var issues
    from dotenv import dotenv_values
    env_vals = dotenv_values()
    api_key = (env_vals.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
               or os.getenv("OPENAI_API_KEY") or "sk-ant-dummy")
    base_url = args.base_url or os.getenv("ANTHROPIC_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    llm_model = args.llm_model

    print(f"  LLM: {llm_model} @ {base_url or 'default Anthropic'}")

    # Reindex if requested
    if args.reindex and os.path.exists(working_dir):
        print(f"  Removing existing index at {working_dir}...")
        shutil.rmtree(working_dir)

    # Create LightRAG instance
    rag = await create_lightrag_instance(working_dir, llm_model, api_key, base_url)

    # Phase 1: Indexing (skip if index exists)
    index_marker = os.path.join(working_dir, ".index_complete")
    if args.eval_only:
        print("\n[Phase 1] Skipping indexing (--eval-only)")
    elif os.path.exists(index_marker):
        print(f"\n[Phase 1] Index already exists at {working_dir} (skip)")
        print(f"         Use --reindex to rebuild")
    else:
        print(f"\n[Phase 1] Offline Indexing")
        os.makedirs(working_dir, exist_ok=True)
        t0 = time.time()
        await run_indexing(rag, articles, batch_size=args.batch_size)
        elapsed = time.time() - t0
        print(f"  Indexing time: {elapsed:.1f}s")
        # Mark index as complete
        with open(index_marker, "w") as f:
            f.write(f"indexed {len(articles)} articles at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Build chunk_id → article_id mapping from LightRAG's text_chunks store
    chunk_to_article = load_chunk_to_article_map(working_dir)
    print(f"  Chunk→article map: {len(chunk_to_article)} entries")

    # Build content-to-article-id index as fallback
    content_index = build_content_to_article_index(articles)
    print(f"  Content index (fallback): {len(content_index)} entries")

    # Phase 2: Online Evaluation
    print(f"\n[Phase 2] Online Evaluation")
    t0 = time.time()
    results, tested, hits_10 = await run_evaluation(
        rag, benchmark_path, args.mode, chunk_to_article, content_index,
        all_article_ids, output_path, start=args.start, limit=args.limit,
        workers=args.workers,
    )
    elapsed = time.time() - t0

    # Final save
    _save_results(output_path, results, benchmark_path, args.mode, tested, hits_10, "complete")

    # Print summary
    non_skipped = [r for r in results if not r.get("skipped")]
    n = len(non_skipped)

    print("\n" + "=" * 55)
    print(f"LightRAG Baseline Results (mode={args.mode})")
    print("=" * 55)
    if n > 0:
        print(f"\n  {'Metric':<12} {'Value':>10}")
        print(f"  {'-'*12} {'-'*10}")
        for k in K_VALUES:
            avg_hit = sum(r.get(f"lightrag_hit@{k}", 0) for r in non_skipped) / n
            print(f"  Hit@{k:<7} {avg_hit*100:>9.1f}%")
        avg_mrr = sum(r.get("lightrag_mrr", 0) for r in non_skipped) / n
        avg_r10 = sum(r.get("lightrag_recall@10", 0) for r in non_skipped) / n
        print(f"  {'MRR':<12} {avg_mrr:>10.4f}")
        print(f"  {'Recall@10':<12} {avg_r10:>10.4f}")
    print(f"\n  Eval time: {elapsed:.1f}s ({elapsed/max(n,1):.1f}s/query)")
    print(f"  Results saved to: {output_path}")

    await rag.finalize_storages()


def main():
    parser = argparse.ArgumentParser(
        description="LightRAG baseline: index legal articles + evaluate on benchmark",
    )
    parser.add_argument("--db-path", default="data/legal_docs.db", help="Legal docs SQLite DB")
    parser.add_argument("--working-dir", default="data/baseline_lightrag_index",
                        help="LightRAG storage directory")
    parser.add_argument("--test-file", default="data/benchmark/hard-200-qa-benchmark.csv",
                        help="Benchmark CSV file")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file")
    parser.add_argument("--mode", default="hybrid", choices=["naive", "local", "global", "hybrid", "mix"],
                        help="LightRAG query mode (default: hybrid)")
    parser.add_argument("--llm-model", default="claude-3-5-haiku-20241022",
                        help="LLM model name for entity extraction and queries")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible API base URL (default: from env)")
    parser.add_argument("--start", type=int, default=1, help="Start from row number")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--batch-size", type=int, default=20, help="Indexing batch size")
    parser.add_argument("-w", "--workers", type=int, default=1, help="Parallel eval workers")
    parser.add_argument("--reindex", action="store_true", help="Force re-indexing from scratch")
    parser.add_argument("--eval-only", action="store_true", help="Skip indexing, evaluate only")

    args = parser.parse_args()

    print("=" * 55)
    print("LightRAG Baseline — Index & Evaluate")
    print("=" * 55)

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
