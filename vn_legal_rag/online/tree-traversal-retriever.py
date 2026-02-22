"""
Tree Traversal Retriever - LLM-guided navigation through document hierarchy.

PageIndex-inspired reasoning-based retrieval for Vietnamese legal documents.
Implements 3-loop approach for multi-document support:
  Loop 0: Forest → Document (select relevant documents when >1 document)
  Loop 1: Document → Chapter (overview with chapter summaries)
  Loop 2: Chapter → Article (detailed selection with article summaries)

Security:
- Query sanitization to prevent prompt injection
"""

import html
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json

from ..types.tree_models import TreeNode, UnifiedForest, NodeType

logger = logging.getLogger(__name__)


@dataclass
class TreeSearchResult:
    """Result from tree traversal search."""
    target_nodes: List[TreeNode] = field(default_factory=list)
    reasoning_path: List[str] = field(default_factory=list)
    confidence: float = 0.0
    contexts: List[str] = field(default_factory=list)
    # Detailed tracking for ablation
    selected_documents: List[str] = field(default_factory=list)
    selected_chapters: List[str] = field(default_factory=list)
    loop0_reasoning: str = ""
    loop1_reasoning: str = ""
    loop2_reasoning: str = ""


class TreeTraversalRetriever:
    """
    LLM-guided tree navigation for legal document retrieval.

    Implements PageIndex-style 3-loop reasoning for multi-document support:
    Loop 0: Pre-filter - Select relevant documents (when >1 document in forest)
            LLM selects documents based on document summaries
    Loop 1: Overview - Present document structure with chapter summaries
            LLM selects relevant chapters based on keywords
    Loop 2: Detail - For each selected chapter, present article summaries
            LLM selects specific articles
    """

    def __init__(
        self,
        forest: UnifiedForest,
        llm_provider: Any,
        article_summaries: Optional[Dict[str, Any]] = None,
        document_summaries: Optional[List[Dict[str, Any]]] = None,
        max_documents: int = 3,
        max_chapters: int = 6,
        max_articles: int = 7,
        confidence_threshold: float = 0.7,
        domain_config: Optional[Any] = None,
        embedding_gen: Optional[Any] = None,
        dual_retriever: Optional[Any] = None,
    ):
        """
        Initialize retriever.

        Args:
            forest: UnifiedForest to navigate
            llm_provider: LLM provider instance
            article_summaries: Dict mapping article_id to summary dict
            document_summaries: List of document summaries for Loop 0 (multi-doc)
            max_documents: Max documents to select in loop 0 (default: 3)
            max_chapters: Max chapters to select in loop 1
            max_articles: Max articles to select in loop 2
            confidence_threshold: Stop when confidence exceeds this
            domain_config: Optional domain-specific configuration
            embedding_gen: Optional embedding generator for semantic scoring
            dual_retriever: Optional DualLevelRetriever for semantic boost in Loop 2
        """
        self.forest = forest
        self.llm_provider = llm_provider
        # Convert article_summaries to dict by article_id if needed
        self.article_summaries = self._normalize_article_summaries(article_summaries)
        self.document_summaries = document_summaries or []
        self.max_documents = max_documents
        self.max_chapters = max_chapters
        self.max_articles = max_articles
        self.confidence_threshold = confidence_threshold
        self.domain_config = domain_config
        self.embedding_gen = embedding_gen
        self._dual_retriever = dual_retriever

    def _sanitize_query(self, query: str) -> str:
        """
        Sanitize query to prevent prompt injection.

        Escapes HTML/XML special characters that could be used to
        break out of prompt structure.

        Args:
            query: Raw user query

        Returns:
            Sanitized query safe for LLM prompts
        """
        return html.escape(query, quote=True)

    def _normalize_article_summaries(self, summaries: Optional[Dict[str, Any]]) -> Dict[str, Dict]:
        """Convert article summaries to dict by article_id."""
        if not summaries:
            return {}

        # If already a dict by article_id
        if isinstance(summaries, dict):
            # Check if it's the new format with "summaries" list
            if "summaries" in summaries and isinstance(summaries["summaries"], list):
                result = {}
                for item in summaries["summaries"]:
                    article_id = item.get("article_id")
                    if article_id:
                        result[article_id] = item
                return result
            # Already in dict format by article_id
            return summaries

        return {}

    def search(self, query: str) -> TreeSearchResult:
        """
        Search forest via 3-loop LLM-guided traversal.

        Loop 0: (Multi-doc only) Select relevant documents
        Loop 1: Document + Chapter overview → select chapters
        Loop 2: Chapter + Article summaries → select articles

        Args:
            query: User query in Vietnamese

        Returns:
            TreeSearchResult with target nodes and reasoning path
        """
        result = TreeSearchResult()

        # Expand query with abbreviations and synonyms
        from importlib import import_module
        _query_analyzer = import_module(".vietnamese-legal-query-analyzer", "vn_legal_rag.online")
        expand_query = _query_analyzer.expand_query
        expanded = expand_query(query)
        search_query = expanded.expanded

        if expanded.abbreviations_found or expanded.synonyms_applied:
            result.reasoning_path.append(
                f"[Expand] Abbrs: {expanded.abbreviations_found}, Synonyms: {expanded.synonyms_applied}"
            )

        # Get all documents
        all_documents = [tree.root for tree in self.forest.trees.values()]
        if not all_documents:
            return result

        # === LOOP 0: Document Selection (Multi-document only) ===
        if len(all_documents) > 1:
            documents, loop0_conf, loop0_reason = self._loop0_select_documents(
                search_query, all_documents
            )
            result.loop0_reasoning = loop0_reason
            result.selected_documents = [doc.metadata.get("so_hieu", doc.name) for doc in documents]
            result.reasoning_path.append(f"[Loop 0] {loop0_reason} (conf: {loop0_conf:.2f})")
        else:
            # Single document - skip Loop 0
            documents = all_documents
            result.selected_documents = [doc.metadata.get("so_hieu", doc.name) for doc in documents]

        # === LOOP 1: Document → Chapter Selection ===
        selected_chapters, loop1_conf, loop1_reason = self._loop1_select_chapters(
            search_query, documents, topic_hints=expanded.topic_hints
        )

        result.loop1_reasoning = loop1_reason
        result.selected_chapters = [ch.name for ch in selected_chapters]
        result.reasoning_path.append(f"[Loop 1] {loop1_reason} (conf: {loop1_conf:.2f})")

        if not selected_chapters:
            return result

        # AUTO-INCLUDE GENERAL PROVISIONS if needed
        all_chapters = []
        for doc in documents:
            for ch in doc.sub_nodes:
                if ch.node_type == NodeType.CHAPTER:
                    all_chapters.append(ch)

        selected_chapters = self._include_general_provisions_if_needed(
            search_query, selected_chapters, all_chapters
        )
        result.selected_chapters = [ch.name for ch in selected_chapters]

        # === LOOP 2: Chapter → Article Selection ===
        all_selected_articles = []
        loop2_reasonings = []
        loop2_confidences = []

        for chapter in selected_chapters:
            articles, loop2_conf, loop2_reason = self._loop2_select_articles(
                search_query, chapter
            )
            all_selected_articles.extend(articles)
            loop2_reasonings.append(f"{chapter.name}: {loop2_reason}")
            loop2_confidences.append(loop2_conf)

        result.loop2_reasoning = " | ".join(loop2_reasonings)
        result.reasoning_path.append(f"[Loop 2] {result.loop2_reasoning}")

        # Deduplicate and limit articles
        seen_ids = set()
        for article in all_selected_articles:
            if article.node_id not in seen_ids and len(result.target_nodes) < self.max_articles:
                seen_ids.add(article.node_id)
                result.target_nodes.append(article)
                result.contexts.append(self._extract_context(article))

        # Compute confidence
        if result.target_nodes:
            avg_loop2_conf = sum(loop2_confidences) / len(loop2_confidences) if loop2_confidences else 0.5
            result.confidence = loop1_conf * 0.45 + avg_loop2_conf * 0.55
            result.confidence = min(result.confidence, 0.95)

        return result

    def _loop0_select_documents(
        self, query: str, all_documents: List[TreeNode]
    ) -> Tuple[List[TreeNode], float, str]:
        """Loop 0: Select relevant documents from forest using LLM."""
        # Build document overview
        doc_overview = []
        doc_id_to_node = {}
        primary_indices = []  # Indices of primary law documents (Luật)

        for i, doc in enumerate(all_documents):
            doc_id = doc.metadata.get("so_hieu", doc.node_id)
            doc_id_to_node[doc_id] = doc

            # Find matching summary
            summary = None
            for s in self.document_summaries:
                if s.get("doc_id") == doc_id or s.get("so_hieu") == doc_id:
                    summary = s
                    break

            if summary:
                doc_info = {
                    "index": i,
                    "doc_id": summary.get("doc_id", doc_id),
                    "name": summary.get("name", doc.name),
                    "domain": summary.get("domain", ""),
                    "scope_preview": summary.get("scope_preview", ""),
                }
            else:
                doc_info = {
                    "index": i,
                    "doc_id": doc_id,
                    "name": doc.name,
                    "domain": "",
                    "scope_preview": doc.content[:200] if doc.content else "",
                }

            doc_overview.append(doc_info)

            # Detect primary law documents (Luật) vs supporting decrees (Nghị định)
            doc_name = doc_info["name"].lower()
            if "luật" in doc_name and "nghị định" not in doc_name:
                primary_indices.append(i)

        if not doc_overview:
            return all_documents, 0.5, "No document summaries available"

        # STRATEGY: Always include primary laws
        selected_indices = set(primary_indices)

        if len(selected_indices) >= self.max_documents:
            selected = [all_documents[i] for i in list(selected_indices)[:self.max_documents]]
            return selected, 0.85, f"Primary laws: {len(selected)}/{len(all_documents)} docs"

        # Filter remaining documents for LLM selection
        remaining_docs = [
            doc_info for doc_info in doc_overview
            if doc_info["index"] not in selected_indices
        ]

        if not remaining_docs or not self.llm_provider:
            selected = [all_documents[i] for i in selected_indices] if selected_indices else all_documents[:self.max_documents]
            return selected, 0.85, f"Primary laws only: {len(selected)} docs"

        slots_remaining = self.max_documents - len(selected_indices)

        # LLM prompt for selecting supporting documents
        sanitized_query = self._sanitize_query(query)
        prompt = f"""<task>Đánh giá xem câu hỏi có cần văn bản hỗ trợ không. KHÔNG giải thích, CHỈ JSON.</task>

<primary_documents_included>
Đã chọn văn bản chính (Luật). Xem xét có cần thêm văn bản hướng dẫn không.
</primary_documents_included>

<supporting_documents>
{json.dumps(remaining_docs, ensure_ascii=False, indent=2)}
</supporting_documents>

<question>{sanitized_query}</question>

<rules>
- Nếu câu hỏi hỏi về QUY ĐỊNH CHUNG (quyền, nghĩa vụ, điều kiện, khái niệm): KHÔNG cần văn bản hỗ trợ
- Nếu câu hỏi hỏi về THỦ TỤC CHI TIẾT (hồ sơ cụ thể, biểu mẫu, quy trình đăng ký): CÓ THỂ cần văn bản hướng dẫn
- Chỉ chọn văn bản hỗ trợ nếu THỰC SỰ cần thiết để trả lời câu hỏi
- Tối đa {slots_remaining} văn bản hỗ trợ
</rules>

<output_format>
{{"selected_indices": [], "confidence": 0.9}}
hoặc nếu cần văn bản hỗ trợ:
{{"selected_indices": [0], "confidence": 0.8}}
</output_format>

JSON:"""

        try:
            response = self.llm_provider.generate(prompt)
            data = self._parse_json_response(response)

            llm_indices = data.get("selected_indices", [])
            for idx in llm_indices:
                if 0 <= idx < len(remaining_docs):
                    original_idx = remaining_docs[idx]["index"]
                    selected_indices.add(original_idx)
                    if len(selected_indices) >= self.max_documents:
                        break

            selected = [all_documents[i] for i in selected_indices]
            confidence = float(data.get("confidence", 0.7))

            primary_count = len(primary_indices)
            supporting_count = len(selected) - primary_count
            reasoning = f"Primary: {primary_count}, Supporting: {supporting_count}"

            return selected, confidence, reasoning

        except Exception as e:
            selected = [all_documents[i] for i in primary_indices] if primary_indices else all_documents[:self.max_documents]
            return selected, 0.5, f"Fallback (primary only): {e}"

    def _loop1_select_chapters(
        self, query: str, documents: List[TreeNode], topic_hints: List[str] = None
    ) -> Tuple[List[TreeNode], float, str]:
        """Loop 1: Select chapters based on document structure overview using LLM."""
        # Build document overview with chapter summaries
        doc_overview = []
        all_chapters = []

        for doc in documents:
            doc_info = {
                "document": doc.name,
                "so_hieu": doc.metadata.get("so_hieu", ""),
                "chapters": []
            }

            for chapter in doc.sub_nodes:
                if chapter.node_type == NodeType.CHAPTER:
                    all_chapters.append(chapter)
                    chapter_info = {
                        "index": len(all_chapters) - 1,
                        "name": chapter.name,
                        "description": chapter.description,  # Contains keywords
                    }
                    doc_info["chapters"].append(chapter_info)

            doc_overview.append(doc_info)

        if not all_chapters:
            # Fallback: Try direct article search for flat document structures
            all_articles = self._collect_articles_from_documents(documents)
            if all_articles:
                logger.info(f"No chapters found - using fallback direct article search ({len(all_articles)} articles)")
                return all_articles[:self.max_chapters], 0.5, "Flat structure - direct articles"
            return [], 0.0, "No chapters found"

    def _collect_articles_from_documents(self, documents: List[TreeNode]) -> List[TreeNode]:
        """Collect articles directly from documents without chapter structure."""
        articles = []
        for doc in documents:
            for child in doc.sub_nodes:
                if child.node_type == NodeType.ARTICLE:
                    articles.append(child)
        return articles[:self.max_articles]

    def _loop1_select_chapters_continued(self, query: str, documents: List[TreeNode], topic_hints: List[str] = None) -> Tuple[List[TreeNode], float, str]:
        """Continuation helper - unused, kept for compatibility."""
        return [], 0.0, "unused"

        # Build topic hint section if available
        hint_section = ""
        if topic_hints:
            hint_section = f"\n\nGỢI Ý NGỮ NGHĨA: {', '.join(topic_hints)}"

        if not self.llm_provider:
            return all_chapters[:self.max_chapters], 0.5, "No LLM - fallback"

        # LLM prompt for chapter selection
        sanitized_query = self._sanitize_query(query)
        prompt = f"""<task>Chọn index chương phù hợp với câu hỏi. KHÔNG giải thích, CHỈ trả về JSON.</task>

<chapters>
{json.dumps(doc_overview, ensure_ascii=False, indent=2)}
</chapters>

<question>{sanitized_query}</question>{hint_section}

<rules>
- Chọn 1-{self.max_chapters} chương phù hợp nhất
- Trả lời ĐÚNG format JSON bên dưới
- KHÔNG hỏi lại, KHÔNG giải thích
</rules>

<output_format>
{{"selected_indices": [0, 1], "confidence": 0.8}}
</output_format>

JSON:"""

        try:
            response = self.llm_provider.generate(prompt)
            data = self._parse_json_response(response)

            indices = data.get("selected_indices", [])
            valid_indices = [i for i in indices if 0 <= i < len(all_chapters)]
            selected = [all_chapters[i] for i in valid_indices[:self.max_chapters]]

            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", f"Selected {len(selected)} chapters")

            # MULTI-CHAPTER EXPANSION: If single chapter with low confidence, explore alternatives
            if len(selected) == 1 and confidence < 0.98:
                selected, confidence, reasoning = self._expand_low_confidence_selection(
                    query, selected, all_chapters, confidence, reasoning, doc_overview
                )

            return selected, confidence, reasoning

        except Exception as e:
            return [all_chapters[0]] if all_chapters else [], 0.3, f"Fallback: {e}"

    def _loop2_select_articles(
        self, query: str, chapter: TreeNode
    ) -> Tuple[List[TreeNode], float, str]:
        """Loop 2: Select articles within a chapter using article summaries and LLM.

        Enhanced with semantic scoring and DualLevel scoring for better article selection.
        """
        articles = []
        article_infos = []

        def collect_articles(node: TreeNode):
            for child in node.sub_nodes:
                if child.node_type == NodeType.ARTICLE:
                    articles.append(child)
                    # Get article summary if available
                    summary = self.article_summaries.get(child.node_id, {})
                    if isinstance(summary, dict):
                        article_info = {
                            "index": len(articles) - 1,
                            "name": child.name,
                            "title": summary.get("article_title", child.name),
                            "keywords": summary.get("keywords", ""),
                        }
                    else:
                        article_info = {
                            "index": len(articles) - 1,
                            "name": child.name,
                            "title": child.name,
                            "keywords": str(summary) if summary else "",
                        }
                    # Add content preview if no keywords
                    if not article_info["keywords"] and child.content:
                        article_info["content_preview"] = child.content[:200]
                    article_infos.append(article_info)
                elif child.node_type == NodeType.SECTION:
                    collect_articles(child)

        collect_articles(chapter)

        if not articles:
            return [], 0.0, "No articles in chapter"

        # SEMANTIC SCORING: Compute embedding similarity for each article
        if self.embedding_gen and article_infos:
            try:
                self._add_semantic_scores(query, article_infos)
            except Exception:
                pass  # Continue without semantic scores if failed

        # DUALLEVEL SCORING: Use KG+embedding scores from DualLevelRetriever
        if self._dual_retriever and articles:
            try:
                self._add_duallevel_scores(query, articles, article_infos)
            except Exception:
                pass  # Continue without dual scores if failed

        if not self.llm_provider:
            return articles[:self.max_articles], 0.5, "No LLM - fallback"

        # Build score hint for LLM prompt
        has_semantic = any("semantic_score" in info for info in article_infos)
        has_dual = any("dual_score" in info for info in article_infos)
        score_hint = ""
        if has_semantic or has_dual:
            hints = []
            if has_semantic:
                hints.append("'semantic_rank'")
            if has_dual:
                hints.append("'dual_rank' (từ KG)")
            rank_fields = " và ".join(hints)
            score_hint = (
                f"\n- Ưu tiên điều có {rank_fields} thấp (1 = phù hợp nhất). "
                "Nếu 2 điều có nội dung tương đương, chọn điều có rank thấp hơn"
            )

        # LLM prompt for article selection
        sanitized_query = self._sanitize_query(query)
        prompt = f"""<task>Chọn index điều luật phù hợp. KHÔNG giải thích, CHỈ JSON.</task>

<chapter>{chapter.name}</chapter>

<articles>
{json.dumps(article_infos, ensure_ascii=False, indent=2)}
</articles>

<question>{sanitized_query}</question>

<rules>
- Chọn 1-{min(self.max_articles, len(articles))} điều trả lời câu hỏi{score_hint}
- KHÔNG hỏi lại, KHÔNG giải thích
</rules>

<output_format>
{{"selected_indices": [0, 1], "confidence": 0.8}}
</output_format>

JSON:"""

        try:
            response = self.llm_provider.generate(prompt)
            data = self._parse_json_response(response)

            indices = data.get("selected_indices", [])
            valid_indices = [i for i in indices if 0 <= i < len(articles)]
            selected = [articles[i] for i in valid_indices[:self.max_articles]]

            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", f"Selected {len(selected)} articles")

            return selected, confidence, reasoning

        except Exception as e:
            return articles[:3], 0.3, f"Fallback: {e}"

    def _add_semantic_scores(
        self, query: str, article_infos: List[Dict[str, Any]]
    ) -> None:
        """Add semantic similarity scores and relative ranks to article_infos.

        Computes embedding similarity between query and article keywords/title.
        Adds both 'semantic_score' (0.0-1.0) and 'semantic_rank' (1 = most similar).

        Args:
            query: User query
            article_infos: List of article info dicts (modified in place)
        """
        import numpy as np

        # Build texts for embedding
        texts_to_embed = []
        for info in article_infos:
            kw = info.get("keywords", "")
            title = info.get("title", info.get("name", ""))
            text = f"{title} {kw}".strip()
            texts_to_embed.append(text if text else "unknown")

        if not texts_to_embed:
            return

        # Generate embeddings
        query_emb = np.array(self.embedding_gen.generate_embeddings([query])[0])
        article_embs = np.array(self.embedding_gen.generate_embeddings(texts_to_embed))

        # Compute cosine similarities
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        article_norms = article_embs / (
            np.linalg.norm(article_embs, axis=1, keepdims=True) + 1e-9
        )
        similarities = np.dot(article_norms, query_norm)

        # Compute relative ranks (1 = most similar)
        sorted_indices = np.argsort(similarities)[::-1]
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(1, len(sorted_indices) + 1)

        # Add scores and ranks to article_infos
        for i, info in enumerate(article_infos):
            score = float(max(0.0, min(1.0, similarities[i])))
            info["semantic_score"] = round(score, 3)
            info["semantic_rank"] = int(ranks[i])

    def _add_duallevel_scores(
        self,
        query: str,
        articles: List[TreeNode],
        article_infos: List[Dict[str, Any]],
    ) -> None:
        """Add DualLevel KG+embedding scores to article_infos.

        Uses DualLevelRetriever to get combined scores from KG relations
        and embedding similarity. Adds 'dual_score' and 'dual_rank' fields.

        Args:
            query: User query
            articles: List of article TreeNodes in current chapter
            article_infos: List of article info dicts (modified in place)
        """
        # Call DualLevelRetriever
        dual_result = self._dual_retriever.retrieve(query, mode="low", max_results=50)

        if not dual_result or not dual_result.final_scores:
            return

        # Build mapping from article node_id to index in article_infos
        node_id_to_idx = {}
        for i, article in enumerate(articles):
            node_id_to_idx[article.node_id] = i

        # Match DualLevel article IDs to our TreeNode IDs
        dual_scores = dual_result.final_scores  # Dict[str, float]
        matched_scores = {}

        for article_id, score in dual_scores.items():
            if article_id in node_id_to_idx:
                matched_scores[article_id] = score
            else:
                # Try partial match (e.g., "d47" matches "59-2020-QH14:d47")
                for node_id in node_id_to_idx:
                    if article_id in node_id or node_id.endswith(f":{article_id}"):
                        matched_scores[node_id] = score
                        break

        if not matched_scores:
            return

        # Compute relative ranks
        sorted_scores = sorted(matched_scores.values(), reverse=True)

        # Add scores and ranks to article_infos
        for node_id, idx in node_id_to_idx.items():
            if node_id in matched_scores:
                score = matched_scores[node_id]
                article_infos[idx]["dual_score"] = round(score, 3)
                try:
                    rank = sorted_scores.index(score) + 1
                    article_infos[idx]["dual_rank"] = rank
                except ValueError:
                    pass

    def _expand_low_confidence_selection(
        self,
        query: str,
        selected: List[TreeNode],
        all_chapters: List[TreeNode],
        confidence: float,
        reasoning: str,
        doc_overview: List[Dict[str, Any]],
    ) -> Tuple[List[TreeNode], float, str]:
        """Expand chapter selection when confidence is low."""
        # Get indices of already-selected chapters
        selected_indices = set()
        for ch in selected:
            for i, all_ch in enumerate(all_chapters):
                if all_ch.node_id == ch.node_id:
                    selected_indices.add(i)
                    break

        # Filter out already-selected chapters
        remaining_chapters = [
            {"index": i, "name": ch.name, "description": ch.description}
            for i, ch in enumerate(all_chapters)
            if i not in selected_indices
        ]

        if not remaining_chapters or not self.llm_provider:
            return selected, confidence, reasoning

        # Ask LLM for alternative chapters
        sanitized_query = self._sanitize_query(query)
        expansion_prompt = f"""<task>Thêm 1 chương liên quan. KHÔNG giải thích, CHỈ JSON.</task>

<selected>{selected[0].name}</selected>
<question>{sanitized_query}</question>

<other_chapters>
{json.dumps(remaining_chapters, ensure_ascii=False, indent=2)}
</other_chapters>

<output_format>
{{"add_chapter_index": 5, "confidence": 0.7}}
hoặc nếu không cần thêm:
{{"add_chapter_index": null, "confidence": 0.9}}
</output_format>

JSON:"""

        try:
            expansion_response = self.llm_provider.generate(expansion_prompt)
            expansion_data = self._parse_json_response(expansion_response)

            add_index = expansion_data.get("add_chapter_index")
            alt_confidence = float(expansion_data.get("confidence", 0.5))

            if add_index is not None and alt_confidence > 0.6:
                if 0 <= add_index < len(all_chapters) and add_index not in selected_indices:
                    alt_chapter = all_chapters[add_index]
                    selected.append(alt_chapter)
                    combined_conf = confidence * 0.6 + alt_confidence * 0.4
                    expanded_reasoning = f"{reasoning} | Expanded: +{alt_chapter.name}"
                    return selected, combined_conf, expanded_reasoning

        except Exception:
            pass

        return selected, confidence, reasoning

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response with logging for failures."""
        import re
        # Try to extract JSON from response
        # Handle markdown code blocks
        response = response.strip()
        if response.startswith("```"):
            # Remove markdown code block markers
            lines = response.split("\n")
            response = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        # Try to find JSON object
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse failed on regex match: {e}. Response: {response[:100]}...")

        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}. Response: {response[:100]}...")
            return {}

    def _include_general_provisions_if_needed(
        self, query: str, selected: List[TreeNode], all_chapters: List[TreeNode]
    ) -> List[TreeNode]:
        """Auto-include general provisions chapter for procedural queries."""
        # Check if general provisions already included
        for ch in selected:
            if "quy định chung" in ch.name.lower():
                return selected

        # Check if query mentions terms that need general definitions
        general_terms = ["khái niệm", "định nghĩa", "giải thích", "là gì", "như thế nào"]
        query_lower = query.lower()
        needs_general = any(term in query_lower for term in general_terms)

        if needs_general:
            for ch in all_chapters:
                if "quy định chung" in ch.name.lower() and ch not in selected:
                    selected.insert(0, ch)
                    break

        return selected

    def _extract_context(self, node: TreeNode) -> str:
        """Extract context from node and ancestors."""
        path = self.forest.get_path_to_root(node.node_id)
        context_parts = []

        if path and path[0].node_type == NodeType.DOCUMENT:
            context_parts.append(f"Văn bản: {path[0].name}")

        breadcrumb = " > ".join([n.name for n in path[1:]])
        if breadcrumb:
            context_parts.append(f"Vị trí: {breadcrumb}")

        context_parts.append(f"\n{node.name}")
        if node.content:
            context_parts.append(node.content)

        return "\n".join(context_parts)


def build_tree_retriever(
    forest: UnifiedForest,
    llm_provider: Any,
    article_summaries: Optional[Dict[str, Any]] = None,
    document_summaries: Optional[List[Dict[str, Any]]] = None,
    domain_config: Optional[Any] = None,
    **kwargs
) -> TreeTraversalRetriever:
    """Convenience function to build retriever."""
    return TreeTraversalRetriever(
        forest=forest,
        llm_provider=llm_provider,
        article_summaries=article_summaries,
        document_summaries=document_summaries,
        domain_config=domain_config,
        **kwargs
    )
