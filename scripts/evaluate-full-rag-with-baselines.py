#!/usr/bin/env python3
"""
Full RAG evaluation with baselines on hard-200 benchmark.

Runs the complete pipeline (retrieval + LLM answer generation) alongside
ablation baselines + real-time traditional baselines (BM25, TF-IDF, Semantic,
Keyword) for each question. All baselines run on the full 840-article database
with doc-qualified IDs.

Usage:
    # Default: hard-200 benchmark
    python scripts/evaluate-full-rag-with-baselines.py -o results/full-rag-200.json

    # Custom test file + limit
    python scripts/evaluate-full-rag-with-baselines.py --test-file data/benchmark/combined-qa-benchmark.csv --limit 10 -o results/test.json

    # Parallel workers
    python scripts/evaluate-full-rag-with-baselines.py -w 2 -o results/full-rag-200.json

    # Select specific ablation baselines (default: tree_only,dual_only)
    python scripts/evaluate-full-rag-with-baselines.py --ablations tree_only,dual_only -o results/out.json

    # Skip traditional baselines (BM25, TF-IDF, etc.)
    python scripts/evaluate-full-rag-with-baselines.py --no-traditional -o results/out.json

    # Convert JSON to CSV
    python scripts/evaluate-full-rag-with-baselines.py --to-csv results/full-rag-200.json
"""

import argparse
import csv
import io
import json
import logging
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress noisy warnings BEFORE importing vn_legal_rag modules
logging.getLogger("vn_legal_rag").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

_stderr = sys.stderr
sys.stderr = io.StringIO()

from dotenv import load_dotenv
load_dotenv()

from vn_legal_rag.online import LegalGraphRAG, create_legal_graphrag
from vn_legal_rag.types import NodeType, UnifiedForest

sys.stderr = _stderr

try:
    from vn_legal_rag import AblationConfig, get_paper_ablation_configs
except ImportError:
    AblationConfig = None
    get_paper_ablation_configs = None

logger = logging.getLogger(__name__)

# Default: no ablation baselines (components are interdependent, ablation misleading)
DEFAULT_ABLATIONS = []

# Traditional baselines to run real-time
TRADITIONAL_BASELINES = ["bm25", "tfidf", "semantic", "keyword"]

K_VALUES = [1, 3, 5, 10, 20, 30]


# Import traditional baselines from separate module
from importlib.util import spec_from_file_location, module_from_spec as _mfs

_spec = spec_from_file_location(
    "traditional_baseline_retrievers",
    str(project_root / "scripts" / "traditional-baseline-retrievers.py"),
)
_trad_mod = _mfs(_spec)
_spec.loader.exec_module(_trad_mod)
init_traditional_baselines = _trad_mod.init_traditional_baselines


# ---------------------------------------------------------------------------
# Article extraction helpers (reused from evaluate-retrieval-performance)
# ---------------------------------------------------------------------------

def extract_expected_article_ids(article_ids: str) -> set:
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


def extract_tree_articles(result) -> list:
    """Extract article IDs from tree search in ranked order."""
    articles, seen = [], set()
    if result.tree_search_result:
        for node in result.tree_search_result.target_nodes:
            if node.node_type == NodeType.ARTICLE and node.node_id and node.node_id not in seen:
                articles.append(node.node_id)
                seen.add(node.node_id)
    return articles


def extract_kg_articles(result) -> list:
    """Extract article IDs from KG/citations in ranked order."""
    articles, seen = [], set()
    if result.citations:
        for cite in result.citations:
            source_id = cite.get("source_id", "")
            article_id = None
            if source_id and ":d" in source_id:
                match = re.match(r'(.+?:d\d+)', source_id)
                if match:
                    article_id = match.group(1)
            if not article_id:
                cite_str = cite.get("citation_string", "")
                num_match = re.search(r"Điều (\d+)", cite_str)
                if num_match:
                    article_id = f"unknown:d{num_match.group(1)}"
            if article_id and article_id not in seen:
                articles.append(article_id)
                seen.add(article_id)
    return articles


def merge_ranked_articles(tree: list, kg: list) -> list:
    """Merge tree + KG articles preserving rank order."""
    seen = set(tree)
    merged = list(tree)
    for a in kg:
        if a not in seen:
            merged.append(a)
            seen.add(a)
    return merged


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
# JSON to CSV converter
# ---------------------------------------------------------------------------

def convert_json_to_csv(json_path: str):
    """Convert flat JSON results to CSV."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data if isinstance(data, list) else data.get("results", [])
    if not rows:
        print("No results found in JSON file.")
        return

    csv_path = json_path.replace(".json", ".csv")
    # Collect all keys across all rows for header
    all_keys = []
    seen_keys = set()
    for row in rows:
        for k in row.keys():
            if k not in seen_keys:
                all_keys.append(k)
                seen_keys.add(k)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            # Flatten list/dict values to strings
            flat = {}
            for k, v in row.items():
                if isinstance(v, (list, dict)):
                    flat[k] = json.dumps(v, ensure_ascii=False)
                else:
                    flat[k] = v
            writer.writerow(flat)

    print(f"CSV saved to: {csv_path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# RAG initialization (shared across configs)
# ---------------------------------------------------------------------------

def init_rag_system(config_path: str, ablation_config=None):
    """Initialize RAG system from config, returns (rag, config)."""
    from vn_legal_rag.utils import load_config, load_kg, load_summaries, build_forest_from_db
    from vn_legal_rag.offline import LegalDocumentDB
    from vn_legal_rag.utils import create_llm_provider, create_embedding_provider

    config = load_config(config_path)

    kg_path = config.get("kg", {}).get("path", "data/kg_enhanced/legal_kg.json")
    chapter_summaries_path = config.get("kg", {}).get("chapter_summaries", "data/kg_enhanced/chapter_summaries.json")
    article_summaries_path = config.get("kg", {}).get("article_summaries", "data/kg_enhanced/article_summaries.json")
    document_summaries_path = config.get("kg", {}).get("document_summaries", "data/kg_enhanced/document_summaries.json")
    domain_groups_path = config.get("kg", {}).get("domain_groups", "data/kg_enhanced/domain_groups.json")
    db_path = config.get("database", {}).get("path", "data/legal_docs.db")

    kg = load_kg(kg_path)
    chapter_summaries = load_summaries(chapter_summaries_path) or {}
    article_summaries = load_summaries(article_summaries_path) or {}
    document_summaries_raw = load_summaries(document_summaries_path) or {}
    domain_groups = load_summaries(domain_groups_path) or {}

    # Convert document_summaries to Loop 0 format
    document_summaries = []
    if isinstance(document_summaries_raw, dict):
        for doc_id, s in document_summaries_raw.items():
            domain = ", ".join(s.get("domain_keywords", [])[:15]) if s.get("domain_keywords") else ""
            document_summaries.append({
                "doc_id": s.get("doc_id", doc_id),
                "so_hieu": s.get("so_hieu", doc_id),
                "name": s.get("ten_van_ban", doc_id),
                "loai": s.get("loai_van_ban", ""),
                "domain": domain,
                "scope_preview": (s.get("scope", ""))[:200],
            })

    forest = build_forest_from_db(db_path, chapter_summaries)
    db = LegalDocumentDB(db_path)

    llm_config = config.get("llm", {})
    llm_kwargs = {
        "provider": llm_config.get("provider", "anthropic"),
        "model": llm_config.get("model", "claude-3-5-haiku-20241022"),
    }
    if llm_config.get("use_cache", True):
        llm_kwargs["cache_db"] = llm_config.get("cache_db", "data/llm_cache.db")
    if llm_config.get("base_url"):
        llm_kwargs["base_url"] = llm_config["base_url"]
    llm_provider = create_llm_provider(**llm_kwargs)
    embedding_gen = create_embedding_provider()

    rag = create_legal_graphrag(
        kg=kg, forest=forest, db=db,
        llm_provider=llm_provider,
        embedding_gen=embedding_gen,
        article_summaries=article_summaries,
        document_summaries=document_summaries,
        domain_groups=domain_groups,
        config=config,
        ablation_config=ablation_config,
    )
    return rag, config, {
        "kg": kg, "forest": forest, "db": db,
        "llm_provider": llm_provider, "embedding_gen": embedding_gen,
        "article_summaries": article_summaries,
        "document_summaries": document_summaries,
        "domain_groups": domain_groups,
        "chapter_summaries": chapter_summaries,
    }


def create_rag_with_ablation(shared_resources: dict, config: dict, ablation_config):
    """Create a new RAG instance reusing shared resources."""
    return create_legal_graphrag(
        kg=shared_resources["kg"],
        forest=shared_resources["forest"],
        db=shared_resources["db"],
        llm_provider=shared_resources["llm_provider"],
        embedding_gen=shared_resources["embedding_gen"],
        article_summaries=shared_resources["article_summaries"],
        document_summaries=shared_resources["document_summaries"],
        domain_groups=shared_resources["domain_groups"],
        config=config,
        ablation_config=ablation_config,
    )


# ---------------------------------------------------------------------------
# Query execution with retry
# ---------------------------------------------------------------------------

def generate_answer_from_articles(question: str, article_ids: list, db, llm_provider):
    """Generate LLM answer from retrieved article IDs using same prompt as RAG."""
    if not article_ids or not llm_provider:
        return ""

    # Fetch article texts and build contexts
    context_parts = []
    for aid in article_ids[:10]:  # limit to top 10 for prompt
        text = ""
        if db and ":" in aid:
            try:
                article = db.get_article_by_id(aid)
                if article:
                    text = article.content or article.raw_text or ""
            except Exception:
                pass
        if not text:
            continue
        # Build reference label
        try:
            doc_id, article_ref = aid.rsplit(":", 1)
            article_num = article_ref[1:] if article_ref.startswith("d") else article_ref
            so_hieu = doc_id.replace("-", "/", 2)
            label = f"Điều {article_num} {so_hieu}"
        except Exception:
            label = aid
        context_parts.append(f"[{len(context_parts)+1}] ({label}) {text}")

    if not context_parts:
        return ""

    context_text = "\n\n".join(context_parts)
    prompt = f"""Bạn là luật sư tư vấn pháp luật Việt Nam.

CÂU HỎI: {question}

TÀI LIỆU THAM KHẢO:
{context_text}

HƯỚNG DẪN:
1. Đọc kỹ tất cả tài liệu trên. Chọn ra những điều luật LIÊN QUAN đến câu hỏi, bỏ qua những điều không liên quan.
2. Dùng những điều luật liên quan để trả lời. Chỉ cần MỘT điều luật liên quan là đủ để trả lời.
3. Trả lời tự nhiên như luật sư, KHÔNG nhắc đến "tài liệu tham khảo" hay "tài liệu được cung cấp".
4. Trích dẫn: "Căn cứ vào Điều X [Tên văn bản]" (VD: "Căn cứ vào Điều 206 Luật Doanh nghiệp 2020").
5. Cuối câu trả lời ghi "Căn cứ pháp lý:" liệt kê các điều đã dùng.
6. CHỈ nói "Xin lỗi, tôi không tìm thấy quy định pháp luật liên quan" khi KHÔNG CÓ BẤT KỲ điều luật nào liên quan.

Trả lời:"""

    try:
        return llm_provider.generate(prompt)
    except Exception:
        return ""


def run_query_with_retry(rag, question: str, max_retries=5, base_delay=5, max_results=30):
    """Run RAG query with rate-limit retry. Returns (result, error)."""
    for attempt in range(max_retries):
        try:
            result = rag.query(question, max_results=max_results, adaptive_retrieval=True)
            return result, None
        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["rate", "limit", "429", "quota"]):
                delay = base_delay * (2 ** attempt)
                print(f"       [Rate limit] Waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            return None, str(e)
    return None, "Max retries exceeded (rate limit)"


# ---------------------------------------------------------------------------
# Process a single question across full RAG + baselines
# ---------------------------------------------------------------------------

def process_question(
    row: dict,
    rag_full,
    rag_baselines: dict,
    traditional_baselines: dict,
    max_articles: int = 30,
    db=None,
    llm_provider=None,
):
    """Process one question: full RAG + ablation baselines + traditional baselines.

    Returns a flat dict suitable for JSON/CSV output.
    """
    stt = row.get("STT", "?")
    question = row.get("Content", "") or row.get("question", "")
    article_ids = row.get("Article_IDs", "") or row.get("article_ids", "")
    category = row.get("Category", "")
    reference_answer = row.get("Câu trả lời", "")
    reference_law = row.get("Điều luật tham chiếu", "")
    expected = extract_expected_article_ids(article_ids)

    record = {
        "stt": stt,
        "category": category,
        "question": question,
        "expected_articles": sorted(expected),
        "num_expected": len(expected),
        "reference_answer": reference_answer,
        "reference_law": reference_law,
    }

    if not expected:
        record["skipped"] = True
        return record

    record["skipped"] = False

    # --- Full RAG (with LLM answer) ---
    result, error = run_query_with_retry(rag_full, question, max_results=max_articles)
    if error:
        record["full_error"] = error
        record["full_answer"] = ""
        record["full_retrieved"] = []
        record["full_hit@10"] = 0
    else:
        tree_arts = extract_tree_articles(result)
        kg_arts = extract_kg_articles(result)
        ranked = merge_ranked_articles(tree_arts, kg_arts)
        ir = calc_ir_metrics(expected, ranked)

        record["full_answer"] = result.response
        record["full_retrieved"] = ranked
        record["full_tree_articles"] = tree_arts
        record["full_kg_articles"] = kg_arts
        record["full_confidence"] = result.confidence
        for k in K_VALUES:
            record[f"full_hit@{k}"] = ir[f"hit@{k}"]
            record[f"full_recall@{k}"] = round(ir[f"recall@{k}"], 4)
        record["full_mrr"] = round(ir["rr"], 4)

    # --- Ablation baselines (retrieval only, no LLM answer) ---
    for bname, rag_b in rag_baselines.items():
        result_b, error_b = run_query_with_retry(rag_b, question, max_results=max_articles)
        if error_b:
            record[f"{bname}_error"] = error_b
            record[f"{bname}_hit@10"] = 0
            continue

        tree_b = extract_tree_articles(result_b)
        kg_b = extract_kg_articles(result_b)
        ranked_b = merge_ranked_articles(tree_b, kg_b)
        ir_b = calc_ir_metrics(expected, ranked_b)

        record[f"{bname}_retrieved"] = ranked_b
        for k in K_VALUES:
            record[f"{bname}_hit@{k}"] = ir_b[f"hit@{k}"]
            record[f"{bname}_recall@{k}"] = round(ir_b[f"recall@{k}"], 4)
        record[f"{bname}_mrr"] = round(ir_b["rr"], 4)

    # --- Traditional baselines (BM25, TF-IDF, Semantic, Keyword) ---
    for tname, retriever in traditional_baselines.items():
        try:
            ranked_t = retriever.search(question, top_k=max_articles)
            ir_t = calc_ir_metrics(expected, ranked_t)

            record[f"{tname}_retrieved"] = ranked_t[:10]  # store top 10 only
            for k in K_VALUES:
                record[f"{tname}_hit@{k}"] = ir_t[f"hit@{k}"]
                record[f"{tname}_recall@{k}"] = round(ir_t[f"recall@{k}"], 4)
            record[f"{tname}_mrr"] = round(ir_t["rr"], 4)
        except Exception as e:
            record[f"{tname}_error"] = str(e)
            record[f"{tname}_hit@10"] = 0

    # --- Generate LLM answer for best traditional baseline ---
    if traditional_baselines and llm_provider and db:
        best_name, best_mrr = None, -1
        for tname in traditional_baselines:
            mrr = record.get(f"{tname}_mrr", 0)
            if mrr > best_mrr:
                best_mrr = mrr
                best_name = tname
        if best_name:
            best_retrieved = record.get(f"{best_name}_retrieved", [])
            if best_retrieved:
                answer = generate_answer_from_articles(question, best_retrieved, db, llm_provider)
                record["best_baseline_name"] = best_name
                record["best_baseline_answer"] = answer

    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full RAG evaluation with baselines (flat JSON output)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config/default.yaml", help="Config file path")
    parser.add_argument("--test-file", default="data/benchmark/hard-200-qa-benchmark.csv",
                        help="Test CSV file (default: hard-200)")
    parser.add_argument("--start", type=int, default=1, help="Start from row number")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file")
    parser.add_argument("--workers", "-w", type=int, default=1, help="Parallel workers")
    parser.add_argument("--ablations", default=",".join(DEFAULT_ABLATIONS),
                        help=f"Comma-separated ablation names (default: {','.join(DEFAULT_ABLATIONS)})")
    parser.add_argument("--no-ablations", action="store_true", help="Skip ablation baselines")
    parser.add_argument("--no-traditional", action="store_true",
                        help="Skip traditional baselines (BM25, TF-IDF, Semantic, Keyword)")
    parser.add_argument("--max-articles", type=int, default=30,
                        help="Max articles retrieved for all methods (default: 30)")
    parser.add_argument("--to-csv", metavar="JSON_FILE", help="Convert existing JSON results to CSV and exit")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # CSV conversion mode
    if args.to_csv:
        convert_json_to_csv(args.to_csv)
        return

    print("=" * 70)
    print("VN LEGAL RAG — Full RAG + Baselines Evaluation")
    print("=" * 70)

    # --- Init full RAG ---
    print("\n[1/5] Initializing full RAG system...")
    rag_full, config, shared = init_rag_system(args.config)
    print("      Full RAG ready")

    # --- Init ablation baselines ---
    ablation_names = [] if args.no_ablations else [b.strip() for b in args.ablations.split(",") if b.strip()]
    rag_baselines = {}

    if ablation_names and get_paper_ablation_configs:
        print(f"\n[2/5] Initializing {len(ablation_names)} ablation baselines...")
        all_ablations = get_paper_ablation_configs()
        for bname in ablation_names:
            if bname in all_ablations:
                rag_baselines[bname] = create_rag_with_ablation(shared, config, all_ablations[bname])
                print(f"      ✓ {bname}")
            else:
                print(f"      ✗ {bname} (not found, skipping)")
    else:
        print("\n[2/5] No ablation baselines selected")

    # --- Init traditional baselines (BM25, TF-IDF, Semantic, Keyword) ---
    traditional = {}
    if not args.no_traditional:
        print(f"\n[3/5] Initializing traditional baselines (real-time on full DB)...")
        db_path = config.get("database", {}).get("path", "data/legal_docs.db")
        # Use a separate embedding provider for semantic baseline
        # to avoid sharing corrupted CUDA state with RAG's embedding model
        baseline_embedding_gen = create_embedding_provider()
        traditional = init_traditional_baselines(
            db_path=db_path,
            embedding_gen=baseline_embedding_gen,
        )
        for method in traditional:
            print(f"      ✓ {method}")
    else:
        print("\n[3/5] Traditional baselines skipped")

    # --- Load test data ---
    print(f"\n[4/5] Loading test data from {args.test_file}...")
    with open(args.test_file, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # Apply start/limit
    tasks = []
    for i, row in enumerate(rows):
        if (i + 1) < args.start:
            continue
        if args.limit and len(tasks) >= args.limit:
            break
        tasks.append(row)

    all_baselines = list(rag_baselines.keys()) + list(traditional.keys())
    print(f"      Questions: {len(tasks)} (start={args.start}, limit={args.limit})")
    print(f"      Max articles per method: {args.max_articles}")
    print(f"      Configs: full + {all_baselines}")

    # --- Run evaluation ---
    print(f"\n[5/5] Running evaluation ({args.workers} worker(s))...")
    print("=" * 70)
    header_baselines = " ".join(f"{b[:8]:>8}" for b in all_baselines)
    print(f"{'#':>4} {'STT':>5} {'Full':>5} {header_baselines}  Running")
    print("-" * 70)

    all_results = []
    stats_lock = Lock()
    completed = 0

    # Aggregate stats
    agg = {"full": defaultdict(int)}
    for b in all_baselines:
        agg[b] = defaultdict(int)

    def process_and_report(row):
        nonlocal completed
        record = process_question(row, rag_full, rag_baselines, traditional,
                                  max_articles=args.max_articles, db=shared["db"], llm_provider=shared["llm_provider"])

        with stats_lock:
            completed += 1
            n = completed

            if record.get("skipped"):
                print(f"[{n:4d}] STT {record['stt']:>5} SKIPPED")
                return record

            # Update aggregate stats
            full_hit = record.get("full_hit@10", 0)
            agg["full"]["tested"] += 1
            agg["full"]["hits"] += full_hit

            baseline_hits = []
            for b in all_baselines:
                bh = record.get(f"{b}_hit@10", 0)
                agg[b]["tested"] += 1
                agg[b]["hits"] += bh
                baseline_hits.append(bh)

            # Print row
            full_str = "HIT" if full_hit else "MISS"
            b_strs = " ".join(
                f"{'HIT':>8}" if h else f"{'MISS':>8}" for h in baseline_hits
            )

            tested = agg["full"]["tested"]
            full_rate = agg["full"]["hits"] / tested * 100

            print(f"[{n:4d}] STT {record['stt']:>5} {full_str:>5} {b_strs}  Hit@10={full_rate:.1f}%")

        return record

    # Incremental save: write partial results after each query
    output_path = args.output if args.output.endswith(".json") else f"{args.output}.json"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    def _save_incremental():
        """Save current results to disk for progress monitoring."""
        sorted_results = sorted(all_results, key=lambda r: int(r.get("stt", 0)))
        inc_data = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "test_file": args.test_file,
                "config": args.config,
                "total_questions": len(tasks),
                "completed": len(all_results),
                "status": "in_progress",
                "max_articles": args.max_articles,
            },
            "results": sorted_results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(inc_data, f, ensure_ascii=False, indent=2)

    try:
        if args.workers == 1:
            for row in tasks:
                record = process_and_report(row)
                all_results.append(record)
                _save_incremental()
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(process_and_report, row): row for row in tasks}
                for future in as_completed(futures):
                    try:
                        record = future.result()
                        all_results.append(record)
                        _save_incremental()
                    except Exception as e:
                        row = futures[future]
                        print(f"[ERR] STT {row.get('STT', '?')} - {e}")
    except KeyboardInterrupt:
        print("\n\n*** INTERRUPTED ***")
        _save_incremental()
        print(f"  Partial results ({len(all_results)} queries) saved to: {output_path}")

    # Sort results by STT
    all_results.sort(key=lambda r: int(r.get("stt", 0)))

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    tested = agg["full"]["tested"]
    if tested > 0:
        # Summary table: Full + all baselines
        full_hits = agg["full"]["hits"]
        print(f"\n  {'Config':<25} {'Hit@10':>8} {'Rate':>8}")
        print(f"  {'-'*25} {'-'*8} {'-'*8}")
        print(f"  {'full (ours)':<25} {full_hits:>5}/{tested:<3} {full_hits/tested*100:>7.1f}%")
        for b in all_baselines:
            bh = agg[b]["hits"]
            bt = agg[b]["tested"]
            if bt > 0:
                print(f"  {b:<25} {bh:>5}/{bt:<3} {bh/bt*100:>7.1f}%")

        # Detailed IR metrics from results
        non_skipped = [r for r in all_results if not r.get("skipped")]
        if non_skipped:
            n_ns = len(non_skipped)
            print(f"\n  Detailed IR Metrics (n={n_ns}):")
            print(f"  {'Config':<18} {'MRR':>7} {'H@1':>6} {'H@3':>6} {'H@5':>6} {'H@10':>6} {'R@10':>7}")
            print(f"  {'-'*18} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")

            # Full RAG
            avg_mrr = sum(r.get("full_mrr", 0) for r in non_skipped) / n_ns
            row_strs = [f"  {'full (ours)':<18} {avg_mrr:>7.4f}"]
            for k in [1, 3, 5, 10]:
                avg_hit = sum(r.get(f"full_hit@{k}", 0) for r in non_skipped) / n_ns
                row_strs.append(f"{avg_hit*100:>5.1f}%")
            avg_r10 = sum(r.get("full_recall@10", 0) for r in non_skipped) / n_ns
            row_strs.append(f"{avg_r10:>7.4f}")
            print(" ".join(row_strs))

            # All baselines
            for b in all_baselines:
                b_mrr = sum(r.get(f"{b}_mrr", 0) for r in non_skipped) / n_ns
                row_strs = [f"  {b:<18} {b_mrr:>7.4f}"]
                for k in [1, 3, 5, 10]:
                    avg_hit = sum(r.get(f"{b}_hit@{k}", 0) for r in non_skipped) / n_ns
                    row_strs.append(f"{avg_hit*100:>5.1f}%")
                avg_r10 = sum(r.get(f"{b}_recall@10", 0) for r in non_skipped) / n_ns
                row_strs.append(f"{avg_r10:>7.4f}")
                print(" ".join(row_strs))

    # --- Save ---
    output_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_file": args.test_file,
            "config": args.config,
            "total_questions": len(tasks),
            "total_tested": tested,
            "ablation_baselines": list(rag_baselines.keys()),
            "traditional_baselines": list(traditional.keys()),
            "max_articles": args.max_articles,
        },
        "results": all_results,
    }

    output_path = args.output if args.output.endswith(".json") else f"{args.output}.json"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"  {len(all_results)} rows (flat JSON)")
    print(f"  Convert to CSV: python {__file__} --to-csv {output_path}")


if __name__ == "__main__":
    main()
