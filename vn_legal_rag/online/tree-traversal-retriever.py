"""
Tree Traversal Retriever - LLM-guided navigation through document hierarchy.

PageIndex-inspired reasoning-based retrieval for Vietnamese legal documents.
Implements 3-loop approach for multi-document support:
  Loop 0: Forest → Document (select relevant documents when >1 document)
  Loop 1: Document → Chapter (overview with chapter summaries)
  Loop 2: Chapter → Article (detailed selection with article summaries)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json
import os

from ..types.tree_models import TreeNode, UnifiedForest, NodeType


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
        domain_groups: Optional[Dict[str, Any]] = None,
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
        self.domain_groups = domain_groups or {}
        self.max_documents = max_documents
        self.max_chapters = max_chapters
        self.max_articles = max_articles
        self.confidence_threshold = confidence_threshold
        self.domain_config = domain_config
        self.embedding_gen = embedding_gen
        self._dual_retriever = dual_retriever

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
        """Loop 0: Select relevant documents by choosing a domain group.

        Uses pre-computed domain groups (from offline phase) to select ALL
        documents in the matching domain. Falls back to LLM individual
        selection if domain groups are unavailable.
        """
        # Build doc_id → TreeNode mapping
        doc_id_map = {}  # maps both so_hieu and node_id to index
        for i, doc in enumerate(all_documents):
            so_hieu = doc.metadata.get("so_hieu", "")
            doc_id_map[so_hieu] = i
            doc_id_map[doc.node_id] = i

        # === Domain group selection (preferred) ===
        groups = self.domain_groups.get("domain_groups", {})
        if groups and len(groups) > 1:
            return self._select_by_domain_group(
                query, all_documents, groups, doc_id_map
            )

        # === Fallback: single domain or no groups → return all docs ===
        return all_documents, 0.7, f"All {len(all_documents)} docs (single domain)"

    def _select_by_domain_group(
        self, query: str, all_documents: List[TreeNode],
        groups: Dict[str, Any], doc_id_map: Dict[str, int],
    ) -> Tuple[List[TreeNode], float, str]:
        """Select documents by matching query to domain group."""
        # Build domain overview for LLM
        domain_overview = []
        group_keys = list(groups.keys())
        for i, (gid, group) in enumerate(groups.items()):
            domain_overview.append({
                "index": i,
                "label": group.get("label", gid),
                "keywords": ", ".join(group.get("domain_keywords", [])[:10]),
                "doc_count": group.get("doc_count", len(group.get("documents", []))),
            })

        # Use LLM to select domain
        if self.llm_provider and len(group_keys) > 1:
            prompt = f"""<task>Chọn lĩnh vực pháp luật phù hợp với câu hỏi. CHỈ JSON.</task>

<domains>
{json.dumps(domain_overview, ensure_ascii=False, indent=2)}
</domains>

<question>{query}</question>

<rules>
- Chọn 1 lĩnh vực phù hợp nhất
- Nếu câu hỏi liên quan nhiều lĩnh vực, chọn lĩnh vực chính
</rules>

<output_format>
{{"selected_index": 0, "confidence": 0.9}}
</output_format>

JSON:"""
            try:
                response = self.llm_provider.generate(prompt)
                data = self._parse_json_response(response)
                selected_idx = data.get("selected_index")
                # Handle both "selected_index" and "selected_indices" formats
                if selected_idx is None:
                    indices = data.get("selected_indices", [])
                    selected_idx = indices[0] if indices else None
                if selected_idx is not None and 0 <= selected_idx < len(group_keys):
                    group_id = group_keys[selected_idx]
                    group = groups[group_id]
                    confidence = float(data.get("confidence", 0.8))
                    return self._resolve_group_docs(
                        query, group, all_documents, doc_id_map, confidence
                    )
            except Exception:
                pass  # Fall through to keyword matching

        # Fallback: keyword matching (no LLM needed)
        return self._select_domain_by_keywords(
            query, groups, group_keys, all_documents, doc_id_map
        )

    def _select_domain_by_keywords(
        self, query: str, groups: Dict[str, Any],
        group_keys: List[str], all_documents: List[TreeNode],
        doc_id_map: Dict[str, int],
    ) -> Tuple[List[TreeNode], float, str]:
        """Fallback domain selection using keyword overlap with query."""
        query_lower = query.lower()
        query_syllables = set(query_lower.split())
        best_group_id = None
        best_score = 0.0
        for gid, group in groups.items():
            keywords = group.get("domain_keywords", [])
            score = 0
            for kw in keywords:
                kw_syllables = set(kw.lower().split())
                overlap = len(query_syllables & kw_syllables)
                if overlap > 0:
                    score += overlap
            if score > best_score:
                best_score = score
                best_group_id = gid
        if best_group_id:
            group = groups[best_group_id]
            return self._resolve_group_docs(
                query, group, all_documents, doc_id_map, 0.6
            )
        # No match — return all documents
        return all_documents, 0.4, f"No domain match, all {len(all_documents)} docs"

    def _resolve_group_docs(
        self, query: str, group: Dict[str, Any], all_documents: List[TreeNode],
        doc_id_map: Dict[str, int], confidence: float,
    ) -> Tuple[List[TreeNode], float, str]:
        """Resolve a domain group's doc IDs to TreeNode list.

        If group has more docs than max_documents, use LLM to narrow down
        to the most relevant ones for the current query.
        """
        group_doc_ids = group.get("documents", [])
        label = group.get("label", "?")[:40]
        selected = []
        for doc_id in group_doc_ids:
            idx = doc_id_map.get(doc_id)
            if idx is not None:
                selected.append(all_documents[idx])
            else:
                # Try matching with normalized so_hieu (slash format)
                normalized = doc_id.replace("-", "/")
                idx = doc_id_map.get(normalized)
                if idx is not None:
                    selected.append(all_documents[idx])
        if not selected:
            return all_documents, 0.4, f"Domain '{label}' no docs found, using all"
        # If group has more docs than max_documents, narrow down with LLM
        if len(selected) > self.max_documents and self.llm_provider:
            narrowed = self._narrow_docs_within_domain(query, selected)
            if narrowed:
                return narrowed, confidence, f"Domain '{label}': {len(selected)} docs → {len(narrowed)} selected"
        return selected, confidence, f"Domain '{label}': {len(selected)} docs"

    def _narrow_docs_within_domain(
        self, query: str, documents: List[TreeNode]
    ) -> Optional[List[TreeNode]]:
        """Narrow documents within a domain group using LLM."""
        doc_list = []
        for i, doc in enumerate(documents):
            so_hieu = doc.metadata.get("so_hieu", doc.name)
            title = doc.metadata.get("title", doc.name)[:80]
            summary_text = ""
            for s in self.document_summaries:
                if s.get("doc_id") == doc.node_id or s.get("so_hieu") == so_hieu:
                    summary_text = s.get("scope_preview", "")[:150]
                    break
            doc_list.append({"index": i, "so_hieu": so_hieu, "title": title, "scope": summary_text})

        prompt = f"""<task>Chọn tối đa {self.max_documents} văn bản phù hợp nhất. CHỈ JSON.</task>

<documents>
{json.dumps(doc_list, ensure_ascii=False, indent=1)}
</documents>

<question>{query}</question>

<output_format>{{"selected_indices": [0, 2, 3]}}</output_format>

JSON:"""
        try:
            response = self.llm_provider.generate(prompt)
            data = self._parse_json_response(response)
            indices = data.get("selected_indices", [])
            if indices:
                selected = [documents[i] for i in indices if 0 <= i < len(documents)]
                if selected:
                    return selected
        except Exception:
            pass
        return None

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
                    }
                    # Ablation: skip description when LOOP1_NO_DESCRIPTION=1
                    if not os.environ.get("LOOP1_NO_DESCRIPTION"):
                        chapter_info["description"] = chapter.description
                    doc_info["chapters"].append(chapter_info)

            doc_overview.append(doc_info)

        if not all_chapters:
            return [], 0.0, "No chapters found"

        # Build topic hint section if available
        hint_section = ""
        if topic_hints:
            hint_section = f"\n\nGỢI Ý NGỮ NGHĨA: {', '.join(topic_hints)}"

        if not self.llm_provider:
            return all_chapters[:self.max_chapters], 0.5, "No LLM - fallback"

        # LLM prompt for chapter selection
        prompt = f"""<task>Chọn index chương phù hợp với câu hỏi. KHÔNG giải thích, CHỈ trả về JSON.</task>

<chapters>
{json.dumps(doc_overview, ensure_ascii=False, indent=2)}
</chapters>

<question>{query}</question>{hint_section}

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
            response = self.llm_provider.generate(prompt, temperature=0.0)
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
            # Fallback: prefer largest chapters (most articles) over first chapter
            fallback = sorted(all_chapters, key=lambda ch: len(ch.sub_nodes), reverse=True)[:self.max_chapters]
            if not fallback:
                fallback = [all_chapters[0]] if all_chapters else []
            return fallback, 0.2, f"Fallback (largest chapters): {e}"

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
        prompt = f"""<task>Chọn index điều luật phù hợp. KHÔNG giải thích, CHỈ JSON.</task>

<chapter>{chapter.name}</chapter>

<articles>
{json.dumps(article_infos, ensure_ascii=False, indent=2)}
</articles>

<question>{query}</question>

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
        expansion_prompt = f"""<task>Thêm 1 chương liên quan. KHÔNG giải thích, CHỈ JSON.</task>

<selected>{selected[0].name}</selected>
<question>{query}</question>

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
        """Parse JSON from LLM response, handling nested objects."""
        import re
        response = response.strip()
        # Remove markdown code block markers
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        # Try direct parse first (handles nested JSON correctly)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Find the outermost JSON object using brace counting
        start = response.find("{")
        if start >= 0:
            depth = 0
            for i in range(start, len(response)):
                if response[i] == "{":
                    depth += 1
                elif response[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(response[start:i + 1])
                        except json.JSONDecodeError:
                            break

        # Fallback: simple non-nested match
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

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
