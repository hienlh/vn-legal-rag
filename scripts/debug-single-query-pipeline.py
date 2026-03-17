#!/usr/bin/env python3
"""
Debug single query through the full RAG pipeline with step-by-step logging.

Shows input/output of each pipeline step:
  Step 1: Query Analysis (intent, type, expansion, keywords)
  Step 2: DualLevel Retrieval (6-component scores)
  Step 3: Tree Traversal Loop 0 (document selection)
  Step 4: Tree Traversal Loop 1 (chapter selection)
  Step 5: Tree Traversal Loop 2 (article selection)
  Step 6: Cross-check Validation (Tree vs DualLevel overlap)
  Step 7: KG Expansion (cross-chapter relations)
  Step 8: RRF Merge (Semantic Bridge fusion)
  Step 9: LLM Response Generation

Usage:
    python scripts/debug-single-query-pipeline.py --query "Điều kiện thành lập CTCP?"
    python scripts/debug-single-query-pipeline.py --query "Phạt bao nhiêu khi vượt đèn đỏ?" --no-llm
    python scripts/debug-single-query-pipeline.py --query "..." --expected "59-2020-QH14:d111,59-2020-QH14:d112"
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vn_legal_rag.utils import (
    load_config, load_kg, load_summaries, build_forest_from_db,
    create_llm_provider, create_embedding_provider,
)
from vn_legal_rag.offline import LegalDocumentDB
from vn_legal_rag.online import LegalGraphRAG, create_legal_graphrag

# Import sub-modules for direct step access
from importlib import import_module

_query_analyzer_mod = import_module(
    ".vietnamese-legal-query-analyzer", "vn_legal_rag.online"
)
_tree_mod = import_module(".tree-traversal-retriever", "vn_legal_rag.online")
_dual_mod = import_module(".dual-level-retriever", "vn_legal_rag.online")
_bridge_mod = import_module(".semantic-bridge-rrf-merger", "vn_legal_rag.online")
_ablation_mod = import_module(
    ".ablation-config-for-rag-component-testing", "vn_legal_rag.types"
)

VietnameseLegalQueryAnalyzer = _query_analyzer_mod.VietnameseLegalQueryAnalyzer
TreeSearchResult = _tree_mod.TreeSearchResult
AblationConfig = _ablation_mod.AblationConfig


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def header(step_num: int, title: str):
    print(f"\n{'='*80}")
    print(f"{BOLD}{BLUE}  STEP {step_num}: {title}{RESET}")
    print(f"{'='*80}")


def sub_header(title: str):
    print(f"\n{BOLD}{CYAN}  >> {title}{RESET}")


def kv(key: str, value, indent: int = 4):
    prefix = " " * indent
    print(f"{prefix}{DIM}{key}:{RESET} {value}")


def table_row(rank: int, article_id: str, score: float, source: str = "",
              hit: bool = None):
    hit_str = ""
    if hit is True:
        hit_str = f" {GREEN}HIT{RESET}"
    elif hit is False:
        hit_str = f" {RED}MISS{RESET}"
    src = f" [{source}]" if source else ""
    print(f"    {rank:>3}. {article_id:<30} score={score:.4f}{src}{hit_str}")


def elapsed(start: float) -> str:
    return f"{time.time() - start:.2f}s"


# ---------------------------------------------------------------------------
# Init system (reuses same pattern as evaluate scripts)
# ---------------------------------------------------------------------------

def init_system(config_path: str):
    """Initialize all RAG components."""
    config = load_config(config_path)

    kg_path = config.get("kg", {}).get("path", "data/kg_enhanced/legal_kg.json")
    chapter_summaries_path = config.get("kg", {}).get(
        "chapter_summaries", "data/kg_enhanced/chapter_summaries.json"
    )
    article_summaries_path = config.get("kg", {}).get(
        "article_summaries", "data/kg_enhanced/article_summaries.json"
    )
    document_summaries_path = config.get("kg", {}).get(
        "document_summaries", "data/kg_enhanced/document_summaries.json"
    )
    domain_groups_path = config.get("kg", {}).get(
        "domain_groups", "data/kg_enhanced/domain_groups.json"
    )
    db_path = config.get("database", {}).get("path", "data/legal_docs.db")

    print(f"{DIM}Loading components...{RESET}")
    kg = load_kg(kg_path)
    chapter_summaries = load_summaries(chapter_summaries_path) or {}
    article_summaries = load_summaries(article_summaries_path) or {}
    document_summaries_raw = load_summaries(document_summaries_path) or {}
    domain_groups = load_summaries(domain_groups_path) or {}

    # Convert document_summaries to Loop 0 format
    document_summaries = []
    if isinstance(document_summaries_raw, dict):
        for doc_id, s in document_summaries_raw.items():
            domain = (
                ", ".join(s.get("domain_keywords", [])[:15])
                if s.get("domain_keywords")
                else ""
            )
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
    )

    print(
        f"{DIM}  KG: {len(kg.get('entities', []))} entities, "
        f"{len(kg.get('relations', []))} relations{RESET}"
    )
    print(f"{DIM}  Forest: {len(forest.trees)} documents{RESET}")
    print(f"{DIM}  Article summaries: {len(article_summaries)} entries{RESET}")
    print(f"{DIM}  Document summaries: {len(document_summaries)} entries{RESET}")
    domains = domain_groups.get("domain_groups", {})
    print(f"{DIM}  Domain groups: {len(domains)} groups{RESET}")
    print(f"{GREEN}  System ready.{RESET}\n")

    return rag, config


# ---------------------------------------------------------------------------
# Debug pipeline
# ---------------------------------------------------------------------------

def debug_query(rag: LegalGraphRAG, query: str, expected: set, skip_llm: bool):
    """Run the pipeline step-by-step with detailed output."""
    total_start = time.time()

    # ------------------------------------------------------------------
    # STEP 1: Query Analysis
    # ------------------------------------------------------------------
    header(1, "Query Analysis")
    t = time.time()
    analyzed = rag.query_analyzer.analyze(query)
    kv("Input", f'"{query}"')
    kv("Intent", analyzed.intent.value)
    kv("Query Type", analyzed.query_type.value)
    kv("Article Refs", analyzed.article_refs or "(none)")
    kv("Law Refs", analyzed.law_refs or "(none)")
    kv("Keywords", ", ".join(analyzed.keywords))
    if analyzed.expanded:
        exp = analyzed.expanded
        kv("Expanded Query", exp.expanded[:200])
        if exp.abbreviations_found:
            kv("Abbreviations", exp.abbreviations_found)
        if exp.synonyms_applied:
            kv("Synonyms", exp.synonyms_applied)
        if exp.topic_hints:
            kv("Topic Hints", exp.topic_hints)
    kv("Time", elapsed(t))

    # ------------------------------------------------------------------
    # STEP 2: DualLevel Retrieval
    # ------------------------------------------------------------------
    header(2, "DualLevel Retrieval (6-component semantic search)")
    t = time.time()
    dual_result = rag.dual_retriever.retrieve(query, mode="low", max_results=50)
    kv("Input", f'query="{query}", mode="low", max_results=50')
    kv("Time", elapsed(t))

    if dual_result and dual_result.final_scores:
        top_dual = sorted(
            dual_result.final_scores.items(), key=lambda x: x[1], reverse=True
        )[:15]
        sub_header(f"Top {len(top_dual)} DualLevel articles (by final_score)")
        for rank, (aid, score) in enumerate(top_dual, 1):
            is_hit = aid in expected if expected else None
            table_row(rank, aid, score, "dual", hit=is_hit)

        # Show score components for top 5
        if dual_result.score_components:
            sub_header("Score breakdown (top 5)")
            for aid, _ in top_dual[:5]:
                comps = dual_result.score_components.get(aid, {})
                if comps:
                    parts = [f"{k}={v:.3f}" for k, v in comps.items() if v > 0]
                    print(f"      {aid}: {', '.join(parts)}")

        # Low-level details
        if dual_result.low_level:
            ll = dual_result.low_level
            sub_header("Low-level sub-scores")
            kv("Entities found", len(ll.entities), indent=6)
            kv("PPR scored", len(ll.ppr_scores), indent=6)
            kv("Semantic scored", len(ll.semantic_scores), indent=6)
            kv("Keyphrase scored", len(ll.keyphrase_scores), indent=6)
            kv("Concept scored", len(ll.concept_scores), indent=6)
    else:
        print(f"    {YELLOW}(no DualLevel results){RESET}")

    # ------------------------------------------------------------------
    # STEP 3: Tree Traversal (all loops)
    # ------------------------------------------------------------------
    header(3, "Tree Traversal (Loop 0 → 1 → 2)")
    t = time.time()
    tree_result = rag.tree_retriever.search(query)
    tree_time = elapsed(t)

    # Loop 0
    sub_header("Loop 0: Document Selection")
    kv("Selected Documents", tree_result.selected_documents or "(all)")
    kv("Reasoning", tree_result.loop0_reasoning or "(skipped)")

    # Loop 1
    sub_header("Loop 1: Chapter Selection")
    kv("Selected Chapters", tree_result.selected_chapters or "(none)")
    kv("Reasoning", tree_result.loop1_reasoning or "(none)")

    # Loop 2
    sub_header("Loop 2: Article Selection")
    kv("Reasoning", tree_result.loop2_reasoning or "(none)")
    if tree_result.target_nodes:
        sub_header(
            f"Tree target articles ({len(tree_result.target_nodes)})"
        )
        for i, node in enumerate(tree_result.target_nodes, 1):
            is_hit = node.node_id in expected if expected else None
            table_row(i, node.node_id, tree_result.confidence, "tree", hit=is_hit)
    else:
        print(f"    {YELLOW}(no tree articles selected){RESET}")

    kv("Tree Confidence", f"{tree_result.confidence:.3f}")
    kv("Reasoning Path", " → ".join(tree_result.reasoning_path) if tree_result.reasoning_path else "(none)")
    kv("Time", tree_time)

    # ------------------------------------------------------------------
    # STEP 4: Cross-check Validation
    # ------------------------------------------------------------------
    header(4, "Cross-check Validation (Tree vs DualLevel)")
    tree_ids = {n.node_id for n in tree_result.target_nodes}
    if dual_result and dual_result.final_scores:
        dual_top = sorted(
            dual_result.final_scores.items(), key=lambda x: x[1], reverse=True
        )
        dual_top5_ids = {aid for aid, _ in dual_top[:5]}
        dual_top10_ids = {aid for aid, _ in dual_top[:10]}
        overlap5 = tree_ids & dual_top5_ids
        overlap10 = tree_ids & dual_top10_ids
        ratio5 = len(overlap5) / max(1, min(len(tree_ids), len(dual_top5_ids)))
        ratio10 = len(overlap10) / max(1, min(len(tree_ids), len(dual_top10_ids)))
        kv("Tree articles", sorted(tree_ids))
        kv("Dual top-5", sorted(dual_top5_ids))
        kv("Overlap (top-5)", f"{len(overlap5)} articles, ratio={ratio5:.2f}")
        kv("Overlap (top-10)", f"{len(overlap10)} articles, ratio={ratio10:.2f}")

        agreement = "HIGH" if ratio5 >= 0.4 else "LOW"
        color = GREEN if ratio5 >= 0.4 else YELLOW
        kv("Agreement", f"{color}{agreement}{RESET}")

        # Articles in dual but not tree (potential misses)
        dual_only = dual_top10_ids - tree_ids
        if dual_only:
            sub_header("In DualLevel top-10 but NOT in Tree (candidates for expansion)")
            for aid in sorted(dual_only):
                score = dual_result.final_scores.get(aid, 0)
                is_hit = aid in expected if expected else None
                table_row(0, aid, score, "dual-only", hit=is_hit)
    else:
        print(f"    {YELLOW}(no dual result for cross-check){RESET}")

    # ------------------------------------------------------------------
    # STEP 5: KG Expansion
    # ------------------------------------------------------------------
    header(5, "KG Expansion (cross-chapter relations)")
    t = time.time()
    kg_results = []
    if tree_result.target_nodes:
        tree_article_ids = [n.node_id for n in tree_result.target_nodes]
        expanded_ids = rag._expand_via_kg_relations(tree_article_ids, query)
        new_ids = set(expanded_ids) - set(tree_article_ids)
        kv("Input (tree articles)", tree_article_ids)
        kv("Expanded IDs", sorted(new_ids) if new_ids else "(none)")
        for aid in list(new_ids)[:5]:
            text = rag._get_article_text_by_source_id(aid)
            if text:
                kg_results.append({
                    "id": aid,
                    "text": text,
                    "metadata": {
                        "source": "kg_expansion",
                        "source_id": aid,
                        "article_id": aid,
                    },
                })
        if kg_results:
            sub_header(f"KG expansion results ({len(kg_results)})")
            for i, ctx in enumerate(kg_results, 1):
                is_hit = ctx["id"] in expected if expected else None
                table_row(i, ctx["id"], 0.0, "kg", hit=is_hit)
    else:
        print(f"    {YELLOW}(no tree articles for KG expansion){RESET}")
    kv("Time", elapsed(t))

    # ------------------------------------------------------------------
    # STEP 6: RRF Merge (Semantic Bridge)
    # ------------------------------------------------------------------
    header(6, "RRF Merge (Semantic Bridge)")
    t = time.time()
    merged = rag.semantic_bridge.merge_tree_dual_results(
        tree_result=tree_result,
        dual_result=dual_result,
        kg_results=kg_results,
        enable_adjacent=True,
    )
    kv("Input sources", f"tree={len(tree_result.target_nodes)}, "
       f"dual={len(dual_result.articles) if dual_result else 0}, "
       f"kg={len(kg_results)}")
    kv("Merged total", len(merged))
    kv("Time", elapsed(t))

    if merged:
        sub_header(f"Top {min(20, len(merged))} merged articles (by RRF score)")
        for rank, ctx in enumerate(merged[:20], 1):
            aid = ctx.get("metadata", {}).get("source_id", ctx.get("id", "?"))
            score = ctx.get("score", 0)
            source = ctx.get("metadata", {}).get("source", "")
            is_hit = aid in expected if expected else None
            table_row(rank, aid, score, source, hit=is_hit)

    # ------------------------------------------------------------------
    # Hit@K summary (if expected provided)
    # ------------------------------------------------------------------
    if expected:
        header(7, "Retrieval Evaluation")
        # From tree
        tree_hits = tree_ids & expected
        kv("Expected articles", sorted(expected))
        kv(f"Tree Hit@{len(tree_ids)}", f"{len(tree_hits)}/{len(expected)} "
           f"({sorted(tree_hits) if tree_hits else 'none'})")

        # From merged (RRF)
        for k in [5, 10, 15, 20]:
            merged_top_k = set()
            for ctx in merged[:k]:
                aid = ctx.get("metadata", {}).get("source_id", ctx.get("id", ""))
                merged_top_k.add(aid)
            hits = merged_top_k & expected
            hit_flag = f"{GREEN}HIT{RESET}" if hits else f"{RED}MISS{RESET}"
            kv(f"RRF Hit@{k}", f"{len(hits)}/{len(expected)} {hit_flag} "
               f"({sorted(hits) if hits else 'none'})")

        # Missing articles
        all_retrieved = set()
        for ctx in merged:
            aid = ctx.get("metadata", {}).get("source_id", ctx.get("id", ""))
            all_retrieved.add(aid)
        missing = expected - all_retrieved
        if missing:
            kv(f"{RED}Missing (not in any result){RESET}", sorted(missing))
    else:
        header(7, "Retrieval Summary (no expected articles provided)")
        kv("Total merged", len(merged))
        kv("Tree articles", len(tree_result.target_nodes))

    # ------------------------------------------------------------------
    # STEP 8: LLM Response (optional)
    # ------------------------------------------------------------------
    if not skip_llm:
        header(8, "LLM Response Generation")
        t = time.time()
        contexts = merged[:50]
        llm_contexts = rag._build_llm_contexts(contexts, tree_result, max_llm=10)
        kv("LLM context articles", len(llm_contexts))
        for i, ctx in enumerate(llm_contexts, 1):
            aid = ctx.get("metadata", {}).get("source_id", ctx.get("id", "?"))
            src = ctx.get("metadata", {}).get("source", "")
            print(f"      {i}. {aid} [{src}]")

        try:
            response, confidence = rag._generate_response(
                query=query, contexts=llm_contexts, analyzed=analyzed,
            )
            kv("Confidence", f"{confidence:.2f}")
            kv("Time", elapsed(t))
            sub_header("Generated Answer")
            print(f"\n{response}\n")
        except Exception as e:
            print(f"    {RED}LLM failed: {e}{RESET}")
    else:
        print(f"\n{DIM}    [LLM skipped with --no-llm]{RESET}")

    # ------------------------------------------------------------------
    # Total time
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"{BOLD}  Total pipeline time: {elapsed(total_start)}{RESET}")
    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Debug single query through full RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--query", required=True, help="Question in Vietnamese")
    parser.add_argument(
        "--expected",
        default="",
        help='Comma-separated expected article IDs (e.g. "59-2020-QH14:d111,59-2020-QH14:d112")',
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Config file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM response generation (faster)",
    )
    args = parser.parse_args()

    expected = set()
    if args.expected:
        expected = {a.strip() for a in args.expected.split(",") if a.strip()}

    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}  DEBUG PIPELINE: Single Query{RESET}")
    print(f"{BOLD}{'='*80}{RESET}")
    print(f"  Query:    {args.query}")
    if expected:
        print(f"  Expected: {sorted(expected)}")
    print()

    rag, config = init_system(args.config)
    debug_query(rag, args.query, expected, skip_llm=args.no_llm)


if __name__ == "__main__":
    main()
