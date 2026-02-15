"""
Legal GraphRAG - 3-Tier Query Engine for Vietnamese Legal Documents

Main entry point for legal question answering using 3-tier retrieval:
- Tier 1: Tree Traversal (LLM-guided navigation)
- Tier 2: DualLevel (6-component semantic search)
- Tier 3: Semantic Bridge (RRF-based fusion)

Features:
- Adaptive retrieval based on query type
- Vietnamese legal citation formatting
- Multi-hop graph traversal for cross-references
- Source provenance tracking
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from importlib import import_module

# Import LLM provider factory
from vn_legal_rag.utils import create_llm_provider

# Import database manager
from vn_legal_rag.offline import LegalDocumentDB

# Import from kebab-case filenames
_tree_retriever = import_module(".tree-traversal-retriever", "vn_legal_rag.online")
TreeTraversalRetriever = _tree_retriever.TreeTraversalRetriever
TreeSearchResult = _tree_retriever.TreeSearchResult

_dual_retriever = import_module(".dual-level-retriever", "vn_legal_rag.online")
DualLevelRetriever = _dual_retriever.DualLevelRetriever
DualLevelConfig = _dual_retriever.DualLevelConfig

_semantic_bridge = import_module(".semantic-bridge-rrf-merger", "vn_legal_rag.online")
SemanticBridge = _semantic_bridge.SemanticBridge

_query_analyzer = import_module(".vietnamese-legal-query-analyzer", "vn_legal_rag.online")
VietnameseLegalQueryAnalyzer = _query_analyzer.VietnameseLegalQueryAnalyzer
AnalyzedQuery = _query_analyzer.AnalyzedQuery
QueryIntent = _query_analyzer.QueryIntent
LegalQueryType = _query_analyzer.LegalQueryType

_ppr = import_module(".personalized-page-rank-for-kg", "vn_legal_rag.online")
PersonalizedPageRank = _ppr.PersonalizedPageRank

# Import AblationConfig
_ablation = import_module(".ablation-config-for-rag-component-testing", "vn_legal_rag.types")
AblationConfig = _ablation.AblationConfig


@dataclass
class GraphRAGResponse:
    """Response from legal GraphRAG query."""
    response: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_path: List[str] = field(default_factory=list)
    confidence: float = 0.0
    query_type: LegalQueryType = LegalQueryType.GENERAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Added for semantica compatibility
    tree_search_result: Optional[Any] = None  # TreeSearchResult
    intent: Optional[QueryIntent] = None


class LegalGraphRAG:
    """
    Legal domain GraphRAG with 3-tier retrieval architecture.

    Provides provenance-aware retrieval with Vietnamese legal citations.

    Example:
        >>> rag = LegalGraphRAG(
        ...     kg=legal_kg,
        ...     forest=document_forest,
        ...     db_path="data/legal_docs.db",
        ...     llm_provider=llm_provider
        ... )
        >>> result = rag.query("Phạt bao nhiêu nếu vi phạm Điều 5?")
        >>> print(result.response)
        >>> for citation in result.citations:
        ...     print(citation["citation_string"])
    """

    def __init__(
        self,
        kg: Dict[str, Any],
        forest: Any,  # UnifiedForest
        db: Optional[Any] = None,  # LegalDocumentDB (backward compat)
        db_path: Optional[str] = None,  # semantica-style: string path
        llm_provider: Optional[Union[Any, str]] = None,  # object or string
        llm_model: Optional[str] = "claude-3-5-haiku-20241022",
        llm_base_url: Optional[str] = "http://127.0.0.1:3456",
        llm_cache_db: Optional[str] = None,  # LLM response cache for speed
        embedding_gen: Optional[Any] = None,
        article_summaries: Optional[Dict[str, Any]] = None,
        document_summaries: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
        ablation_config: Optional[AblationConfig] = None,  # semantica-style
    ):
        """
        Initialize LegalGraphRAG.

        Supports both APIs:
        - vn_legal_rag style: db (object), llm_provider (object)
        - semantica style: db_path (string), llm_provider/llm_model/llm_base_url (strings)

        Args:
            kg: Knowledge graph from offline extraction
            forest: UnifiedForest for tree-based retrieval
            db: Optional database object for article text retrieval
            db_path: Optional database path (string, semantica-style)
            llm_provider: LLM provider instance or string (openai/anthropic)
            llm_model: LLM model name (semantica-style)
            llm_base_url: LLM API base URL (semantica-style)
            llm_cache_db: Path to LLM response cache database for faster repeat queries
            embedding_gen: Embedding generator
            article_summaries: Article summaries for Loop 2
            document_summaries: Document summaries for Loop 0
            config: Optional configuration dict
            ablation_config: AblationConfig for ablation studies (semantica-style)
        """
        self.kg = kg
        self.forest = forest

        # Handle database: prefer db object, fallback to db_path
        if db is not None:
            self.db = db
        elif db_path is not None:
            self.db = LegalDocumentDB(db_path)
        else:
            self.db = None

        # Handle LLM provider: string or object
        if isinstance(llm_provider, str):
            self.llm_provider = create_llm_provider(
                provider=llm_provider,
                model=llm_model,
                base_url=llm_base_url,
                cache_db=llm_cache_db,
            )
        else:
            self.llm_provider = llm_provider

        self.embedding_gen = embedding_gen
        self.config = config or {}

        # Store ablation config
        self.ablation_config = ablation_config or AblationConfig()

        # Initialize query analyzer
        self.query_analyzer = VietnameseLegalQueryAnalyzer()

        # Initialize PPR if embedding generator available
        self.ppr = None
        if self.embedding_gen:
            self.ppr = PersonalizedPageRank(kg=kg, embedding_gen=embedding_gen)
            self.ppr.build_node_embeddings()

        # Initialize DualLevel retriever
        dual_config = DualLevelConfig()
        self.dual_retriever = DualLevelRetriever(
            kg=kg,
            embedding_gen=embedding_gen,
            config=dual_config,
            ppr=self.ppr,
        )

        # Initialize Tree retriever
        self.tree_retriever = TreeTraversalRetriever(
            forest=forest,
            llm_provider=self.llm_provider,  # Use the created object, not the string
            article_summaries=article_summaries,
            document_summaries=document_summaries,
            embedding_gen=embedding_gen,
            dual_retriever=self.dual_retriever,
        )

        # Initialize Semantic Bridge (use self.db which may come from db_path)
        self.semantic_bridge = SemanticBridge(kg=kg, db=self.db)

        # Initialize Cross-Encoder Reranker (Stage 2 reranking)
        self._reranker = None
        enable_reranker = self.ablation_config.enable_reranker if hasattr(self.ablation_config, 'enable_reranker') else True
        if enable_reranker:
            try:
                _reranker_mod = import_module(".cross-encoder-reranker-for-legal-documents", "vn_legal_rag.online")
                CrossEncoderReranker = _reranker_mod.CrossEncoderReranker
                self._reranker = CrossEncoderReranker()
            except (ImportError, AttributeError):
                pass  # Reranker not available

    def query(
        self,
        query: str,
        max_results: int = 10,
        adaptive_retrieval: bool = True,
    ) -> GraphRAGResponse:
        """
        Query legal knowledge with natural language.

        Uses 3-tier retrieval:
        - Tier 1: Tree traversal for structured navigation
        - Tier 2: DualLevel for global semantic search
        - Tier 3: Semantic Bridge for result fusion

        Args:
            query: Natural language question in Vietnamese
            max_results: Maximum contexts to retrieve
            adaptive_retrieval: Use query-type-based retrieval strategy

        Returns:
            GraphRAGResponse with answer and citations
        """
        # Step 1: Analyze query
        analyzed = self.query_analyzer.analyze(query)

        # Step 2: Run 3-tier retrieval
        contexts, tree_result = self._retrieve_contexts(
            query, analyzed, max_results, adaptive_retrieval
        )

        # Step 3: Generate response with LLM
        if self.llm_provider:
            response, confidence = self._generate_response(
                query=query,
                contexts=contexts,
                analyzed=analyzed,
            )
        else:
            # Fallback without LLM
            response = self._format_contexts(contexts)
            confidence = 0.8 if contexts else 0.0

        # Step 4: Build response
        return GraphRAGResponse(
            response=response,
            citations=self._extract_citations(contexts),
            reasoning_path=tree_result.reasoning_path if tree_result else [],
            confidence=confidence,
            query_type=analyzed.query_type,
            metadata={
                "query_type": analyzed.query_type.value,
                "intent": analyzed.intent.value,
                "contexts_retrieved": len(contexts),
                "tree_confidence": tree_result.confidence if tree_result else 0.0,
                "query_analyzed": {
                    "keywords": analyzed.keywords,
                    "article_refs": analyzed.article_refs,
                },
                "retrieval_strategy": {
                    "method": "hybrid",
                    "hybrid_alpha": 0.7,
                    "max_hops": 2,
                    "use_temporal": False,
                },
                "ontology_expansion": [],  # Placeholder
            },
            tree_search_result=tree_result,  # semantica-style
            intent=analyzed.intent,  # semantica-style
        )

    def _retrieve_contexts(
        self,
        query: str,
        analyzed: AnalyzedQuery,
        max_results: int,
        adaptive: bool,
    ) -> tuple:
        """
        Retrieve contexts using 3-tier approach with cross-validation.

        Respects ablation_config for component toggling:
        - enable_tree: Tier 1 tree traversal
        - enable_dual_level: Tier 2 DualLevel retrieval
        - enable_semantic_bridge: Tier 3 semantic bridge
        - enable_kg_expansion: KG-based cross-chapter expansion
        - enable_adjacent_expansion: Adjacent article expansion

        Cross-validation logic (from semantica):
        - Compare Tree vs DualLevel results
        - Expand with disagreeing chapters when overlap is low
        - Add high-scoring DualLevel articles that Tree missed

        Returns:
            Tuple of (contexts, tree_result)
        """
        cfg = self.ablation_config
        tree_result = None
        dual_result = None

        # Step 1: DualLevel retrieval FIRST (global semantic search)
        if cfg.enable_dual_level:
            dual_result = self.dual_retriever.retrieve(
                query, mode="low", max_results=30
            )

        # Step 2: Tree retrieval (structured navigation)
        if cfg.enable_tree:
            tree_result = self.tree_retriever.search(query)
        else:
            tree_result = TreeSearchResult()  # Empty result

        # Step 3: Cross-check validation (Tree vs DualLevel)
        if tree_result and tree_result.target_nodes and dual_result and dual_result.final_scores:
            tree_article_ids = {node.node_id for node in tree_result.target_nodes}

            # Get top articles from dual results
            dual_top_articles = sorted(
                dual_result.final_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            dual_article_ids = {aid for aid, _ in dual_top_articles[:5]}

            # Calculate overlap
            overlap = len(tree_article_ids & dual_article_ids)
            overlap_ratio = overlap / max(1, min(len(tree_article_ids), len(dual_article_ids)))

            # Low agreement (<40%): expand with disagreeing chapters
            if overlap_ratio < 0.4:
                tree_chapters = self._get_chapters_from_articles(tree_result.target_nodes)
                dual_chapters = self._get_chapters_from_article_ids([aid for aid, _ in dual_top_articles])
                new_chapters = dual_chapters - tree_chapters

                if new_chapters:
                    self._expand_to_new_chapters(
                        tree_result, dual_top_articles, new_chapters, max_additions=3
                    )

            # Semantic Bridge: Add high-scoring DualLevel articles that Tree missed
            if cfg.enable_semantic_bridge:
                score_threshold = 0.55
                tree_article_set = {node.node_id for node in tree_result.target_nodes}

                for aid, score in dual_top_articles[:7]:
                    if aid not in tree_article_set and score > score_threshold:
                        article_text = self._get_article_text_by_source_id(aid)
                        if article_text:
                            tree_result.contexts.append(article_text)

        # Step 4: KG relation expansion (cross-chapter coverage)
        if cfg.enable_kg_expansion and tree_result and tree_result.target_nodes:
            tree_article_ids = [node.node_id for node in tree_result.target_nodes]
            expanded_ids = self._expand_via_kg_relations(tree_article_ids, query)

            # Add expanded articles
            new_article_ids = set(expanded_ids) - set(tree_article_ids)
            for article_id in list(new_article_ids)[:3]:
                article_text = self._get_article_text_by_source_id(article_id)
                if article_text:
                    tree_result.contexts.append(article_text)

        # Step 5: Merge results using Semantic Bridge
        kg_results = []

        # Case 1: Tree result available - merge tree + dual + KG
        if tree_result and (tree_result.target_nodes or tree_result.contexts):
            if cfg.enable_semantic_bridge:
                merged = self.semantic_bridge.merge_tree_dual_results(
                    tree_result=tree_result,
                    dual_result=dual_result,
                    kg_results=kg_results,
                    enable_adjacent=cfg.enable_adjacent_expansion,
                )
            else:
                # Simple concatenation without bridge
                merged = []
                for node in tree_result.target_nodes:
                    merged.append({
                        "id": node.node_id,
                        "text": node.content,
                        "metadata": {"source": "tree"},
                        "score": tree_result.confidence,
                    })
                if dual_result and hasattr(dual_result, 'results'):
                    for item in dual_result.results:
                        merged.append({
                            "id": item.get("id", ""),
                            "text": item.get("text", ""),
                            "metadata": {"source": "dual_level"},
                            "score": item.get("score", 0.5),
                        })

        # Case 2: No tree but DualLevel available - use DualLevel as primary
        # Handles ablation configs like no_tree, dual_only
        elif dual_result and dual_result.articles:
            dual_contexts = []
            for i, article in enumerate(dual_result.articles[:max_results]):
                article_id = article.get("article_id", "")
                if not article_id:
                    continue
                article_text = self._get_article_text_by_source_id(article_id)
                if article_text:
                    # Get score from final_scores if available
                    score = 0.5
                    if dual_result.final_scores and article_id in dual_result.final_scores:
                        score = dual_result.final_scores[article_id]
                    dual_contexts.append({
                        "id": f"dual-{i}",
                        "text": article_text,
                        "metadata": {
                            "source": "dual",
                            "source_id": article_id,
                            "article_id": article_id,
                        },
                        "score": score,
                    })

            # Merge with KG results if semantic bridge enabled
            if cfg.enable_semantic_bridge and kg_results:
                merged = self.semantic_bridge.merge_dual_and_kg_results(
                    dual_contexts, kg_results
                )
            else:
                merged = dual_contexts if dual_contexts else kg_results

        # Case 3: Fallback to KG results only
        else:
            merged = kg_results

        # Apply cross-encoder reranking if available
        merged = self._apply_reranking(query, merged, max_results)

        # Limit to max_results
        contexts = merged[:max_results]

        return contexts, tree_result

    def _generate_response(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        analyzed: AnalyzedQuery,
    ) -> tuple:
        """
        Generate response using LLM.

        Returns:
            Tuple of (response_text, confidence)
        """
        # Build prompt with contexts
        context_text = "\n\n".join([
            f"[{i+1}] {ctx['text']}"
            for i, ctx in enumerate(contexts)
        ])

        prompt = f"""Bạn là trợ lý pháp lý chuyên về luật Việt Nam.
Trả lời câu hỏi dựa trên các điều khoản pháp luật được cung cấp.
Luôn trích dẫn nguồn bằng số trong ngoặc vuông [1], [2], v.v.

Câu hỏi: {query}

Điều khoản pháp luật:
{context_text}

Trả lời:"""

        try:
            response = self.llm_provider.generate(prompt)
            confidence = 0.85
            return response, confidence
        except Exception:
            return "Không thể tạo câu trả lời.", 0.0

    def _get_article_text(self, article_id: str) -> str:
        """Get article text from database."""
        if self.db:
            return self.semantic_bridge._get_article_text(article_id)
        return ""

    def _extract_citations(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract citations from contexts."""
        citations = []
        for ctx in contexts:
            metadata = ctx.get("metadata", {})
            source_id = metadata.get("source_id", "")
            if source_id:
                citations.append({
                    "source_id": source_id,
                    "citation_string": f"Điều {source_id.split(':d')[-1] if ':d' in source_id else source_id}",
                })
        return citations

    def _format_contexts(self, contexts: List[Dict[str, Any]]) -> str:
        """Format contexts as simple text response."""
        if not contexts:
            return "Không tìm thấy thông tin liên quan."

        formatted = []
        for i, ctx in enumerate(contexts, 1):
            formatted.append(f"[{i}] {ctx['text'][:200]}...")

        return "\n\n".join(formatted)

    def _apply_reranking(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Apply cross-encoder reranking if available.

        Stage 2 reranking using CrossEncoderReranker for higher precision.

        Args:
            query: User query string
            candidates: List of candidate contexts
            top_k: Number of top results to return

        Returns:
            Reranked list of contexts
        """
        # Check if reranker is available
        if not hasattr(self, '_reranker') or not self._reranker:
            return candidates
        if not candidates:
            return candidates

        try:
            # Check if reranker is initialized and available
            if not self._reranker.is_available():
                return candidates

            result = self._reranker.rerank(query, candidates, top_k=top_k)
            if result.reranked_items:
                return result.reranked_items
        except Exception:
            pass

        return candidates

    # =========================================================================
    # Cross-validation and expansion methods (ported from semantica)
    # =========================================================================

    def _get_chapters_from_articles(self, nodes: List) -> set:
        """
        Extract chapter names from tree result nodes.

        Uses the node's path to find the chapter level.

        Args:
            nodes: List of TreeNode objects

        Returns:
            Set of chapter names
        """
        chapters = set()
        for node in nodes:
            # Try to get chapter from node path
            path = getattr(node, "path", None)
            if path and len(path) > 1:
                chapter_name = path[1] if len(path) > 1 else None
                if chapter_name:
                    chapters.add(chapter_name)
                    continue

            # Alternative: extract from node_id using forest path lookup
            node_id = getattr(node, "node_id", "")
            if ":" in node_id and self.tree_retriever:
                try:
                    forest_path = self.tree_retriever.forest.get_path_to_root(node_id)
                    for p in forest_path:
                        if hasattr(p, "node_type") and str(p.node_type).endswith("CHAPTER"):
                            chapters.add(p.name)
                            break
                except Exception:
                    pass

        return chapters

    def _get_chapters_from_article_ids(self, article_ids: List[str]) -> set:
        """
        Extract chapter names from article IDs using forest path lookup.

        Args:
            article_ids: List of article IDs (e.g., "59-2020-QH14:d5")

        Returns:
            Set of chapter names
        """
        chapters = set()
        if not self.tree_retriever:
            return chapters

        for aid in article_ids:
            try:
                forest_path = self.tree_retriever.forest.get_path_to_root(aid)
                for p in forest_path:
                    if hasattr(p, "node_type") and str(p.node_type).endswith("CHAPTER"):
                        chapters.add(p.name)
                        break
            except Exception:
                pass

        return chapters

    def _expand_to_new_chapters(
        self,
        tree_result,
        dual_top_articles: List[tuple],
        new_chapters: set,
        max_additions: int = 3,
    ) -> int:
        """
        Add top articles from new chapters to tree result.

        Args:
            tree_result: TreeSearchResult to expand
            dual_top_articles: List of (article_id, score) from DualLevel
            new_chapters: Set of chapter names to add from
            max_additions: Max articles to add

        Returns:
            Number of articles added
        """
        added = 0
        existing_ids = {node.node_id for node in tree_result.target_nodes}

        for aid, score in dual_top_articles:
            if added >= max_additions:
                break
            if aid in existing_ids:
                continue

            # Check if this article is from a new chapter
            article_chapters = self._get_chapters_from_article_ids([aid])
            if article_chapters & new_chapters:
                article_text = self._get_article_text_by_source_id(aid)
                if article_text:
                    tree_result.contexts.append(article_text)
                    added += 1

        return added

    def _expand_via_kg_relations(
        self, tree_articles: List[str], query: str
    ) -> List[str]:
        """
        Use KG relations to expand article set for multi-topic queries.

        Prioritizes expansion to articles in DIFFERENT chapters than those
        already selected (cross-chapter linking).

        Args:
            tree_articles: List of article IDs from tree traversal
            query: User query for semantic relevance filtering

        Returns:
            Expanded list of article IDs (including original + related)
        """
        if not tree_articles or not self.kg:
            return tree_articles

        expanded = set(tree_articles)
        relations = self.kg.get("relations", []) or self.kg.get("relationships", [])

        # Get chapters of currently selected articles
        selected_chapters = self._get_chapters_from_article_ids(tree_articles)

        # Strong relations for cross-chapter linking
        STRONG_REL_TYPES = {"REFERENCES", "REQUIRES", "DEPENDS_ON", "IMPLEMENTS"}
        EXPANSION_REL_TYPES = {"DEFINED_AS", "CONDITION_FOR", "RELATED_TO", "INCLUDES"}

        cross_chapter_additions = []
        same_chapter_additions = []

        for article_id in tree_articles:
            related = self._get_kg_related_articles(article_id, relations)
            for rel_article, rel_type in related:
                if rel_article in expanded:
                    continue

                rel_chapters = self._get_chapters_from_article_ids([rel_article])
                is_cross_chapter = bool(rel_chapters - selected_chapters)

                if rel_type in STRONG_REL_TYPES and is_cross_chapter:
                    if self._is_semantically_relevant(rel_article, query):
                        cross_chapter_additions.append(rel_article)
                elif rel_type in EXPANSION_REL_TYPES:
                    if self._is_semantically_relevant(rel_article, query):
                        if is_cross_chapter:
                            cross_chapter_additions.append(rel_article)
                        else:
                            same_chapter_additions.append(rel_article)

        # Prioritize cross-chapter (limit 3), then same-chapter (limit 2)
        for aid in cross_chapter_additions[:3]:
            expanded.add(aid)
        for aid in same_chapter_additions[:2]:
            expanded.add(aid)

        return list(expanded)

    def _get_kg_related_articles(
        self, article_id: str, relations: List[Dict]
    ) -> List[tuple]:
        """
        Get articles related to given article via KG relations.

        Args:
            article_id: Source article ID
            relations: List of KG relation dicts

        Returns:
            List of (related_article_id, relation_type) tuples
        """
        related = []

        for rel in relations:
            src = rel.get("source", "") or rel.get("head", "")
            tgt = rel.get("target", "") or rel.get("tail", "")
            rel_type = rel.get("type", "") or rel.get("relation_type", "")

            if article_id in src or src in article_id:
                tgt_article = self._extract_article_id_from_entity(tgt)
                if tgt_article:
                    related.append((tgt_article, rel_type))

            if article_id in tgt or tgt in article_id:
                src_article = self._extract_article_id_from_entity(src)
                if src_article:
                    related.append((src_article, rel_type))

        return related

    def _extract_article_id_from_entity(self, entity_name: str) -> Optional[str]:
        """
        Extract article source_id from KG entity name.

        Args:
            entity_name: Entity name or ID

        Returns:
            Article source_id or None
        """
        if not entity_name:
            return None

        # If already in source_id format
        if ":" in entity_name and entity_name.split(":")[-1].startswith("d"):
            return entity_name

        # Try to find matching entity in KG
        entities = self.kg.get("entities", [])
        for entity in entities:
            name = entity.get("name", "")
            if name == entity_name or entity_name in name:
                metadata = entity.get("metadata", {})
                source_id = metadata.get("source_id")
                if source_id:
                    return source_id

        return None

    def _is_semantically_relevant(self, article_id: str, query: str) -> bool:
        """
        Check if article is semantically relevant to query using embeddings.

        Args:
            article_id: Article source_id
            query: User query

        Returns:
            True if article is relevant (similarity > threshold)
        """
        if not self.embedding_gen:
            return True  # Default to include if no embeddings

        try:
            import numpy as np

            article_text = self._get_article_text_by_source_id(article_id)
            if not article_text:
                return False

            # Use first 500 chars for efficiency
            article_text = article_text[:500]

            # Generate embeddings
            query_emb = np.array(self.embedding_gen.generate_embeddings([query])[0])
            article_emb = np.array(
                self.embedding_gen.generate_embeddings([article_text])[0]
            )

            # Cosine similarity
            similarity = np.dot(query_emb, article_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(article_emb) + 1e-9
            )

            return similarity > 0.4

        except Exception:
            return True  # Default to include on error

    def _get_article_text_by_source_id(self, source_id: str) -> str:
        """
        Get article text by source_id (e.g., '59-2020-QH14:d5').

        Args:
            source_id: Source ID in format 'doc_id:dN'

        Returns:
            Article content text or empty string if not found
        """
        if not source_id or ":" not in source_id:
            return ""

        try:
            doc_id, article_ref = source_id.rsplit(":", 1)
            if article_ref.startswith("d"):
                article_num = int(article_ref[1:])
                # Try semantic_bridge first
                if self.semantic_bridge:
                    return self.semantic_bridge._get_article_text(source_id)
                # Fallback to direct DB query
                if self.db:
                    from sqlalchemy.orm import Session
                    with Session(self.db.engine) as session:
                        from vn_legal_rag.offline import LegalArticleModel
                        article = session.query(LegalArticleModel).filter(
                            LegalArticleModel.so_dieu == article_num,
                            LegalArticleModel.document_id.like(f"%{doc_id}%")
                        ).first()
                        if article:
                            return article.noi_dung or ""
        except (ValueError, AttributeError):
            pass

        return ""

    # =========================================================================
    # Adaptive Threshold & Ambiguity Calibration (ported from semantica)
    # =========================================================================

    def _compute_adaptive_threshold(self, overlap_ratio: float) -> float:
        """
        Compute adaptive score threshold based on Tree-DualLevel agreement.

        When Tree and DualLevel agree strongly (high overlap), use stricter threshold.
        When they disagree (low overlap), use looser threshold to include more candidates.

        Args:
            overlap_ratio: Overlap ratio between Tree and DualLevel results (0.0-1.0)

        Returns:
            Threshold value between 0.4 (loose) and 0.6 (strict)
        """
        # High agreement (>70%) → strict threshold (0.6)
        # Low agreement (<30%) → loose threshold (0.4)
        # Linear interpolation between
        MIN_THRESHOLD = 0.4
        MAX_THRESHOLD = 0.6

        if overlap_ratio >= 0.7:
            return MAX_THRESHOLD
        elif overlap_ratio <= 0.3:
            return MIN_THRESHOLD
        else:
            # Linear interpolation: map 0.3-0.7 → 0.4-0.6
            normalized = (overlap_ratio - 0.3) / 0.4  # 0.0 to 1.0
            return MIN_THRESHOLD + normalized * (MAX_THRESHOLD - MIN_THRESHOLD)

    def _compute_ambiguity_calibration(
        self,
        query: str,
        tree_result: Optional[Any] = None,
        analyzed: Optional[AnalyzedQuery] = None,
    ) -> float:
        """
        Compute calibration factor based on query ambiguity.

        Ambiguous queries (vague, multi-topic) need more candidates.
        Specific queries (article refs, single topic) need fewer, higher-quality.

        Args:
            query: User query string
            tree_result: TreeSearchResult (optional, for confidence info)
            analyzed: AnalyzedQuery (optional, for query analysis info)

        Returns:
            Calibration factor between 0.7 (ambiguous) and 1.0 (specific)
        """
        calibration = 1.0
        ambiguity_signals = 0

        # Signal 1: Query length (very short = ambiguous)
        words = query.split()
        if len(words) <= 3:
            ambiguity_signals += 1

        # Signal 2: No specific article references
        if analyzed:
            if not analyzed.article_refs:
                ambiguity_signals += 1
            # General query type = more ambiguous
            if analyzed.query_type == LegalQueryType.GENERAL:
                ambiguity_signals += 1

        # Signal 3: Low tree confidence = harder to navigate
        if tree_result:
            confidence = getattr(tree_result, "confidence", 1.0)
            if confidence < 0.5:
                ambiguity_signals += 1

        # Signal 4: Question words without specifics
        vague_patterns = ["là gì", "như thế nào", "có những gì", "bao gồm"]
        query_lower = query.lower()
        if any(p in query_lower for p in vague_patterns):
            ambiguity_signals += 1

        # Map signals to calibration (0-4 signals → 1.0-0.7)
        # Each signal reduces calibration by 0.075
        calibration = max(0.7, 1.0 - ambiguity_signals * 0.075)

        return calibration

    def _apply_threshold_calibration(
        self,
        scores: Dict[str, float],
        threshold: float,
        calibration: float,
    ) -> Dict[str, float]:
        """
        Apply adaptive threshold and calibration to scores.

        Args:
            scores: Article ID -> score mapping
            threshold: Base score threshold
            calibration: Calibration factor (0.7-1.0)

        Returns:
            Filtered scores above adjusted threshold
        """
        # Adjust threshold by calibration
        # Lower calibration (ambiguous) → lower effective threshold
        adjusted_threshold = threshold * calibration

        return {
            aid: score
            for aid, score in scores.items()
            if score >= adjusted_threshold
        }


def create_legal_graphrag(
    kg: Dict[str, Any],
    forest: Any,
    db: Optional[Any] = None,
    llm_provider: Optional[Any] = None,
    **kwargs
) -> LegalGraphRAG:
    """Factory function to create LegalGraphRAG instance."""
    return LegalGraphRAG(
        kg=kg,
        forest=forest,
        db=db,
        llm_provider=llm_provider,
        **kwargs
    )
