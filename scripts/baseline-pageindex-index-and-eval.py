#!/usr/bin/env python3
"""
PageIndex baseline: offline indexing + online evaluation on hard-200 benchmark.

Phase 1 (Offline): Convert legal articles to markdown, build semantic tree via PageIndex.
Phase 2 (Online): For each query, LLM reasons over tree structure to find relevant articles.

PageIndex is a vectorless, reasoning-based RAG that uses document structure (tree) for
retrieval instead of vector similarity. It maps well to hierarchical legal documents.

Usage:
    # Full pipeline (index if needed + evaluate)
    python scripts/baseline-pageindex-index-and-eval.py -o results/baseline_pageindex.json

    # Re-index from scratch
    python scripts/baseline-pageindex-index-and-eval.py --reindex -o results/baseline_pageindex.json

    # Evaluate only (skip indexing, tree must exist)
    python scripts/baseline-pageindex-index-and-eval.py --eval-only -o results/baseline_pageindex.json

    # Limit questions
    python scripts/baseline-pageindex-index-and-eval.py --limit 10 -o results/baseline_pageindex_10.json

Dependencies:
    cd baselines/PageIndex && pip install -r requirements.txt
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add PageIndex to path
pageindex_root = project_root / "baselines" / "PageIndex"
sys.path.insert(0, str(pageindex_root))

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

def load_articles_from_db(db_path: str) -> dict:
    """Load all articles grouped by document_id.

    Returns: {
        doc_id: {
            "title": "...",
            "chapters": {
                chapter_id: {
                    "title": "...",
                    "articles": [{article_id, article_number, title, content}, ...]
                }
            }
        }
    }
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Load documents
    cur.execute("SELECT id, title FROM legal_documents")
    docs = {}
    for doc_id, title in cur.fetchall():
        docs[doc_id] = {"title": title, "chapters": {}}

    # Load chapters
    cur.execute("SELECT id, document_id, chapter_number, title FROM legal_chapters ORDER BY document_id, position")
    for ch_id, doc_id, ch_num, ch_title in cur.fetchall():
        if doc_id in docs:
            docs[doc_id]["chapters"][ch_id] = {
                "chapter_number": ch_num,
                "title": ch_title or f"Chương {ch_num}",
                "articles": [],
            }

    # Load articles
    cur.execute("""
        SELECT id, article_number, title, content, document_id, chapter_id
        FROM legal_articles
        ORDER BY document_id, position, article_number
    """)
    all_article_ids = set()
    for art_id, art_num, art_title, content, doc_id, ch_id in cur.fetchall():
        all_article_ids.add(art_id)
        article = {
            "article_id": art_id,
            "article_number": art_num,
            "title": art_title or f"Điều {art_num}",
            "content": content or "",
        }
        if doc_id in docs and ch_id and ch_id in docs[doc_id]["chapters"]:
            docs[doc_id]["chapters"][ch_id]["articles"].append(article)
        elif doc_id in docs:
            # Article without chapter — put in a default chapter
            default_ch = f"{doc_id}:c_default"
            if default_ch not in docs[doc_id]["chapters"]:
                docs[doc_id]["chapters"][default_ch] = {
                    "chapter_number": "0",
                    "title": "Các điều khoản",
                    "articles": [],
                }
            docs[doc_id]["chapters"][default_ch]["articles"].append(article)

    conn.close()
    return docs, all_article_ids


# ---------------------------------------------------------------------------
# Phase 1: Build Markdown + Tree Index
# ---------------------------------------------------------------------------

def generate_markdown_per_document(docs: dict, output_dir: str) -> dict:
    """Generate markdown file per legal document, preserving hierarchy.

    Returns: {doc_id: md_path}
    """
    os.makedirs(output_dir, exist_ok=True)
    md_paths = {}

    for doc_id, doc_info in docs.items():
        md_lines = [f"# {doc_info['title']}\n"]

        for ch_id, ch_info in doc_info["chapters"].items():
            if not ch_info["articles"]:
                continue
            md_lines.append(f"\n## Chương {ch_info['chapter_number']}. {ch_info['title']}\n")

            for article in ch_info["articles"]:
                # Use article_id as an anchor for later mapping
                md_lines.append(
                    f"\n### Điều {article['article_number']}. {article['title']} "
                    f"[{article['article_id']}]\n"
                )
                md_lines.append(f"\n{article['content']}\n")

        md_path = os.path.join(output_dir, f"{doc_id}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        md_paths[doc_id] = md_path

    return md_paths


def build_tree_for_document(md_path: str, model: str, api_key: str, base_url: str = None) -> dict:
    """Build PageIndex tree from a markdown file.

    Uses PageIndex's md_to_tree for structured documents.
    """
    import asyncio

    # Set up env for PageIndex (uses CHATGPT_API_KEY)
    os.environ["CHATGPT_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url

    try:
        from pageindex.page_index_md import md_to_tree

        tree = asyncio.run(md_to_tree(
            md_path=md_path,
            if_thinning=False,  # Keep all nodes for legal precision
            model=model,
            if_add_node_summary="yes",
            if_add_doc_description="no",
            if_add_node_text="yes",  # Need text for retrieval
            if_add_node_id="yes",
        ))
        return tree
    except Exception as e:
        print(f"  [WARN] PageIndex md_to_tree failed for {md_path}: {e}")
        # Fallback: build tree manually from markdown structure
        return build_tree_manually(md_path, doc_id=os.path.basename(md_path).replace(".md", ""))


def build_tree_manually(md_path: str, doc_id: str = "") -> dict:
    """Fallback: build a simple tree from markdown headings without LLM.

    Args:
        md_path: Path to markdown file
        doc_id: Document ID prefix to ensure unique node_ids across documents
    """
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Use doc_id as prefix to prevent node_id collisions across documents
    prefix = f"{doc_id}_" if doc_id else ""

    lines = content.split("\n")
    root = {"title": "", "node_id": f"{prefix}0000", "nodes": [], "text": ""}
    current_chapter = None
    current_article = None
    article_text_lines = []
    node_counter = 1

    def flush_article():
        nonlocal current_article, article_text_lines
        if current_article and article_text_lines:
            current_article["text"] = "\n".join(article_text_lines)
            article_text_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("# ") and not stripped.startswith("## "):
            root["title"] = stripped[2:].strip()
        elif stripped.startswith("## "):
            flush_article()
            current_chapter = {
                "title": stripped[3:].strip(),
                "node_id": f"{prefix}{node_counter:04d}",
                "nodes": [],
                "text": "",
                "summary": stripped[3:].strip(),
            }
            node_counter += 1
            root["nodes"].append(current_chapter)
            current_article = None
        elif stripped.startswith("### "):
            flush_article()
            current_article = {
                "title": stripped[4:].strip(),
                "node_id": f"{prefix}{node_counter:04d}",
                "nodes": [],
                "text": "",
                "summary": "",  # filled after flush
            }
            node_counter += 1
            article_text_lines = []
            if current_chapter:
                current_chapter["nodes"].append(current_article)
            else:
                root["nodes"].append(current_article)
        else:
            if current_article is not None:
                article_text_lines.append(line)

    flush_article()

    # Generate summaries from article text (first 2 sentences, max 150 chars)
    _generate_heuristic_summaries(root)
    return root


def _generate_heuristic_summaries(node: dict):
    """Generate short summaries from article text content (no LLM needed)."""
    text = node.get("text", "").strip()
    if text and not node.get("summary", "").strip():
        # Take first 1-2 meaningful sentences as summary
        # Split by sentence-ending patterns
        sentences = re.split(r'(?<=[.;:])\s+', text)
        summary_parts = []
        total_len = 0
        for s in sentences:
            s = s.strip()
            if not s or len(s) < 5:
                continue
            # Skip clause numbers like "1.", "a)"
            if re.match(r'^\d+\.\s*$', s) or re.match(r'^[a-z]\)\s*$', s):
                continue
            summary_parts.append(s)
            total_len += len(s)
            if total_len >= 120 or len(summary_parts) >= 2:
                break
        if summary_parts:
            node["summary"] = " ".join(summary_parts)[:200]

    for child in node.get("nodes", []):
        _generate_heuristic_summaries(child)


def build_node_to_article_mapping(tree: dict) -> dict:
    """Build mapping from node_id to article_id by parsing node titles.

    Article titles contain [article_id] anchors from markdown generation.
    """
    mapping = {}

    def traverse(node):
        title = node.get("title", "")
        # Extract article_id from "[doc_id:dXX]" anchor in title
        match = re.search(r'\[([^\]]+:d\d+)\]', title)
        if match:
            mapping[node["node_id"]] = match.group(1)
        for child in node.get("nodes", []):
            traverse(child)

    traverse(tree)
    return mapping


# ---------------------------------------------------------------------------
# Phase 2: Online Evaluation via Tree Search
# ---------------------------------------------------------------------------

def _call_llm(prompt: str, model: str, api_key: str, base_url: str = None) -> str:
    """Call LLM via Anthropic API (raw httpx to avoid SDK Bearer header bug)."""
    import httpx

    url = f"{base_url}/v1/messages" if base_url else "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
        "stream": True,
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = httpx.post(url, json=body, headers=headers, timeout=120)
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
            result = "".join(text_parts)
            if result:
                return result
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    return ""


def tree_search_with_llm(question: str, trees: list, node_to_article: dict,
                         model: str, api_key: str, base_url: str = None,
                         max_articles: int = 30) -> list:
    """2-step tree search following PageIndex methodology.

    Step 1: Send document + chapter structure (with summaries) → LLM selects relevant chapters
    Step 2: Send articles within selected chapters (with summaries) → LLM selects articles

    This avoids sending the full 840-article tree in one prompt.
    """
    try:
        # Step 1: Document + Chapter selection
        selected_chapters = _step1_select_chapters(question, trees, model, api_key, base_url)

        if not selected_chapters:
            return [], None

        # Step 2: Article selection within selected chapters
        ranked = _step2_select_articles(
            question, trees, selected_chapters, node_to_article,
            model, api_key, base_url, max_articles,
        )

        return ranked, None

    except Exception as e:
        return [], str(e)


def _step1_select_chapters(question: str, trees: list, model: str,
                           api_key: str, base_url: str = None) -> list:
    """Step 1: LLM selects relevant chapters from document-level tree."""
    # Build compact document → chapter tree (no articles)
    tree_lines = []
    for doc_tree in trees:
        doc_title = doc_tree.get("title", "Untitled")
        tree_lines.append(f"📄 {doc_title}")
        for ch in doc_tree.get("nodes", []):
            nid = ch.get("node_id", "?")
            ch_title = ch.get("title", "")
            ch_summary = ch.get("summary", "")
            n_articles = len(ch.get("nodes", []))
            tree_lines.append(f"  [{nid}] {ch_title} ({n_articles} điều)")
            if ch_summary and ch_summary != ch_title and len(ch_summary) > 10:
                tree_lines.append(f"    Tóm tắt: {ch_summary[:200]}")

    tree_text = "\n".join(tree_lines)

    prompt = f"""Bạn là chuyên gia pháp luật Việt Nam. Nhiệm vụ: chọn các chương liên quan đến câu hỏi.

CÂU HỎI: {question}

DANH SÁCH VĂN BẢN VÀ CHƯƠNG:
{tree_text}

Chọn TẤT CẢ các chương có thể chứa điều luật liên quan. Ưu tiên recall cao (không bỏ sót).
Trả về JSON:
{{
    "thinking": "<lý do>",
    "chapter_ids": ["node_id_1", "node_id_2", ...]
}}"""

    content = _call_llm(prompt, model, api_key, base_url)
    result = parse_json_response(content)
    return result.get("chapter_ids", [])


def _step2_select_articles(question: str, trees: list, selected_chapter_ids: list,
                           node_to_article: dict, model: str, api_key: str,
                           base_url: str = None, max_articles: int = 30) -> list:
    """Step 2: LLM selects specific articles from within selected chapters."""
    selected_set = set(str(c) for c in selected_chapter_ids)

    # Collect articles from selected chapters only
    article_lines = []
    for doc_tree in trees:
        doc_title = doc_tree.get("title", "Untitled")
        doc_has_articles = False
        for ch in doc_tree.get("nodes", []):
            if ch.get("node_id") not in selected_set:
                continue
            if not doc_has_articles:
                article_lines.append(f"\n📄 {doc_title}")
                doc_has_articles = True
            article_lines.append(f"  📁 {ch.get('title', '')}")
            for art in ch.get("nodes", []):
                nid = art.get("node_id", "?")
                art_title = art.get("title", "")
                art_summary = art.get("summary", "")
                article_lines.append(f"    [{nid}] {art_title}")
                if art_summary:
                    article_lines.append(f"      Tóm tắt: {art_summary[:200]}")

    if not article_lines:
        return []

    articles_text = "\n".join(article_lines)

    # Truncate if still too large
    if len(articles_text) > 80000:
        articles_text = articles_text[:80000] + "\n... (truncated)"

    prompt = f"""Bạn là chuyên gia pháp luật Việt Nam. Nhiệm vụ: chọn các điều luật cụ thể trả lời câu hỏi.

CÂU HỎI: {question}

CÁC ĐIỀU LUẬT TRONG CHƯƠNG LIÊN QUAN:
{articles_text}

HƯỚNG DẪN:
1. Đọc kỹ câu hỏi, xác định chủ đề pháp lý.
2. Chọn các điều luật TRỰC TIẾP trả lời hoặc liên quan mật thiết đến câu hỏi.
3. Sắp xếp theo mức độ liên quan giảm dần.
4. Chọn tối đa {max_articles} điều.

Trả về JSON:
{{
    "thinking": "<lý do>",
    "node_list": ["node_id_1", "node_id_2", ...]
}}"""

    content = _call_llm(prompt, model, api_key, base_url)
    result = parse_json_response(content)
    node_ids = result.get("node_list", [])

    # Build reverse map: article_id -> article_id (LLM sometimes returns article IDs
    # like "1393-QD-DHQG:d2" instead of node IDs like "1393-QD-DHQG_0003")
    article_id_set = set(node_to_article.values())

    # Map node_ids to article_ids
    ranked = []
    seen = set()
    for nid in node_ids:
        nid_str = str(nid)
        if nid_str in node_to_article:
            # Proper node_id -> article_id mapping
            aid = node_to_article[nid_str]
        elif nid_str in article_id_set:
            # LLM returned article_id directly (e.g. "1393-QD-DHQG:d2")
            aid = nid_str
        else:
            continue
        if aid not in seen:
            ranked.append(aid)
            seen.add(aid)
        if len(ranked) >= max_articles:
            break

    return ranked


def format_tree_for_prompt(node: dict, depth: int = 0) -> str:
    """Format tree node for LLM prompt, showing structure + summaries."""
    indent = "  " * depth
    node_id = node.get("node_id", "?")
    title = node.get("title", "Untitled")
    summary = node.get("summary", "")

    parts = [f"{indent}[{node_id}] {title}"]
    if summary and len(summary) > 10:
        # Truncate long summaries
        short_summary = summary[:200] + "..." if len(summary) > 200 else summary
        parts.append(f"{indent}  Tóm tắt: {short_summary}")

    children = node.get("nodes", [])
    for child in children:
        parts.append(format_tree_for_prompt(child, depth + 1))

    return "\n".join(parts)


def parse_json_response(content: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    # Remove markdown code block markers
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```\s*', '', content)
    content = content.strip()

    # Try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Brace-counting parser for nested JSON
    start = content.find('{')
    if start == -1:
        return {"node_list": []}

    depth = 0
    end = start
    for i in range(start, len(content)):
        if content[i] == '{':
            depth += 1
        elif content[i] == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    try:
        return json.loads(content[start:end])
    except json.JSONDecodeError:
        # Last resort: extract node_list via regex
        node_ids = re.findall(r'"(\d{4})"', content)
        return {"node_list": node_ids}


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(benchmark_path: str, trees: list, node_to_article: dict,
                   model: str, api_key: str, base_url: str,
                   output_path: str, start: int = 1, limit: int = None):
    """Run evaluation on benchmark dataset."""

    with open(benchmark_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    tasks = []
    for i, row in enumerate(rows):
        if (i + 1) < start:
            continue
        if limit and len(tasks) >= limit:
            break
        tasks.append(row)

    print(f"\n  Evaluating {len(tasks)} questions...")
    print(f"  {'#':>4} {'STT':>5} {'Result':>6}  Running Hit@10")
    print("-" * 55)

    all_results = []
    hits_10 = 0
    tested = 0

    for idx, row in enumerate(tasks):
        stt = row.get("STT", "?")
        question = row.get("Content", "") or row.get("question", "")
        article_ids = row.get("Article_IDs", "") or row.get("article_ids", "")
        category = row.get("Category", "")
        expected = extract_expected_article_ids(article_ids)

        record = {
            "stt": stt,
            "category": category,
            "question": question,
            "expected_articles": sorted(expected),
            "num_expected": len(expected),
        }

        if not expected:
            record["skipped"] = True
            all_results.append(record)
            print(f"[{idx+1:4d}] STT {stt:>5} SKIPPED")
            continue

        record["skipped"] = False
        ranked, error = tree_search_with_llm(
            question, trees, node_to_article, model, api_key, base_url,
        )

        if error:
            record["pageindex_error"] = error
            record["pageindex_retrieved"] = []
            for k in K_VALUES:
                record[f"pageindex_hit@{k}"] = 0
                record[f"pageindex_recall@{k}"] = 0.0
            record["pageindex_mrr"] = 0.0
        else:
            ir = calc_ir_metrics(expected, ranked)
            record["pageindex_retrieved"] = ranked[:30]
            for k in K_VALUES:
                record[f"pageindex_hit@{k}"] = ir[f"hit@{k}"]
                record[f"pageindex_recall@{k}"] = round(ir[f"recall@{k}"], 4)
            record["pageindex_mrr"] = round(ir["rr"], 4)

        all_results.append(record)
        tested += 1
        hit10 = record.get("pageindex_hit@10", 0)
        hits_10 += hit10
        rate = hits_10 / tested * 100

        result_str = "HIT" if hit10 else "MISS"
        print(f"[{idx+1:4d}] STT {stt:>5} {result_str:>6}  {rate:.1f}%")

        # Incremental save
        _save_results(output_path, all_results, benchmark_path, tested, hits_10, "in_progress")

    return all_results, tested, hits_10


def _save_results(output_path: str, results: list, benchmark_path: str,
                  tested: int, hits_10: int, status: str):
    """Save results to JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    non_skipped = [r for r in results if not r.get("skipped")]
    n = len(non_skipped)

    summary = {}
    if n > 0:
        summary = {
            "hit@10": f"{hits_10}/{tested} ({hits_10/tested*100:.1f}%)" if tested else "N/A",
            "mrr": round(sum(r.get("pageindex_mrr", 0) for r in non_skipped) / n, 4),
        }
        for k in K_VALUES:
            avg_hit = sum(r.get(f"pageindex_hit@{k}", 0) for r in non_skipped) / n
            summary[f"hit@{k}_rate"] = round(avg_hit * 100, 1)

    data = {
        "metadata": {
            "baseline": "PageIndex",
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

def main():
    parser = argparse.ArgumentParser(
        description="PageIndex baseline: build tree index + evaluate on benchmark",
    )
    parser.add_argument("--db-path", default="data/legal_docs.db", help="Legal docs SQLite DB")
    parser.add_argument("--index-dir", default="data/baseline_pageindex_index",
                        help="Directory for markdown + tree index files")
    parser.add_argument("--test-file", default="data/benchmark/hard-200-qa-benchmark.csv",
                        help="Benchmark CSV file")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file")
    parser.add_argument("--llm-model", default="claude-3-5-haiku-20241022",
                        help="LLM model for tree building and search")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible API base URL (default: from env)")
    parser.add_argument("--start", type=int, default=1, help="Start from row number")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--reindex", action="store_true", help="Force re-indexing from scratch")
    parser.add_argument("--eval-only", action="store_true", help="Skip indexing, evaluate only")
    parser.add_argument("--use-pageindex-llm", action="store_true",
                        help="Use PageIndex LLM for tree building (requires CHATGPT_API_KEY)")
    parser.add_argument("--manual-tree", action="store_true", default=True,
                        help="Build tree manually without LLM (default: True, faster)")
    parser.add_argument("--no-manual-tree", dest="manual_tree", action="store_false",
                        help="Use PageIndex LLM-based tree building")

    args = parser.parse_args()

    print("=" * 55)
    print("PageIndex Baseline — Index & Evaluate")
    print("=" * 55)

    db_path = args.db_path
    index_dir = args.index_dir
    benchmark_path = args.test_file
    output_path = args.output if args.output.endswith(".json") else f"{args.output}.json"

    # Re-read .env to ensure key is loaded (shell env may have empty ANTHROPIC_API_KEY)
    from dotenv import dotenv_values
    env_vals = dotenv_values()
    api_key = (env_vals.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
               or os.getenv("OPENAI_API_KEY") or "sk-ant-dummy")
    base_url = args.base_url or os.getenv("ANTHROPIC_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    model = args.llm_model

    print(f"  LLM: {model} @ {base_url or 'default OpenAI'}")

    # Load articles
    docs, all_article_ids = load_articles_from_db(db_path)
    total_articles = sum(
        len(a) for d in docs.values() for ch in d["chapters"].values() for a in [ch["articles"]]
    )
    print(f"  Loaded {total_articles} articles from {len(docs)} documents")

    # Phase 1: Build tree index
    tree_cache_path = os.path.join(index_dir, "trees.json")
    md_dir = os.path.join(index_dir, "markdown")

    if args.eval_only and os.path.exists(tree_cache_path):
        print(f"\n[Phase 1] Loading cached trees from {tree_cache_path}")
        with open(tree_cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        trees = cached["trees"]
        node_to_article = cached["node_to_article"]
    elif args.reindex or not os.path.exists(tree_cache_path):
        print(f"\n[Phase 1] Offline Indexing")
        t0 = time.time()

        # Generate markdown
        print("  Generating markdown files...")
        md_paths = generate_markdown_per_document(docs, md_dir)
        print(f"  Generated {len(md_paths)} markdown files")

        # Build trees
        print("  Building tree indices...")
        trees = []
        node_to_article = {}

        for doc_id, md_path in md_paths.items():
            if args.manual_tree:
                tree = build_tree_manually(md_path, doc_id=doc_id)
            else:
                tree = build_tree_for_document(md_path, model, api_key, base_url)

            trees.append(tree)
            mapping = build_node_to_article_mapping(tree)
            node_to_article.update(mapping)
            n_articles = len(mapping)
            print(f"    {doc_id}: {n_articles} article nodes")

        elapsed = time.time() - t0
        print(f"  Tree building time: {elapsed:.1f}s")
        print(f"  Total article nodes mapped: {len(node_to_article)}")

        # Cache trees
        os.makedirs(index_dir, exist_ok=True)
        with open(tree_cache_path, "w", encoding="utf-8") as f:
            json.dump({"trees": trees, "node_to_article": node_to_article},
                      f, ensure_ascii=False, indent=2)
        print(f"  Trees cached to {tree_cache_path}")
    else:
        print(f"\n[Phase 1] Loading cached trees from {tree_cache_path}")
        with open(tree_cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        trees = cached["trees"]
        node_to_article = cached["node_to_article"]

    print(f"  Trees: {len(trees)} documents, {len(node_to_article)} article nodes")

    # Phase 2: Online Evaluation
    print(f"\n[Phase 2] Online Evaluation")
    t0 = time.time()
    results, tested, hits_10 = run_evaluation(
        benchmark_path, trees, node_to_article,
        model, api_key, base_url,
        output_path, start=args.start, limit=args.limit,
    )
    elapsed = time.time() - t0

    # Final save
    _save_results(output_path, results, benchmark_path, tested, hits_10, "complete")

    # Print summary
    non_skipped = [r for r in results if not r.get("skipped")]
    n = len(non_skipped)

    print("\n" + "=" * 55)
    print("PageIndex Baseline Results")
    print("=" * 55)
    if n > 0:
        print(f"\n  {'Metric':<12} {'Value':>10}")
        print(f"  {'-'*12} {'-'*10}")
        for k in K_VALUES:
            avg_hit = sum(r.get(f"pageindex_hit@{k}", 0) for r in non_skipped) / n
            print(f"  Hit@{k:<7} {avg_hit*100:>9.1f}%")
        avg_mrr = sum(r.get("pageindex_mrr", 0) for r in non_skipped) / n
        avg_r10 = sum(r.get("pageindex_recall@10", 0) for r in non_skipped) / n
        print(f"  {'MRR':<12} {avg_mrr:>10.4f}")
        print(f"  {'Recall@10':<12} {avg_r10:>10.4f}")
    print(f"\n  Eval time: {elapsed:.1f}s ({elapsed/max(n,1):.1f}s/query)")
    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
