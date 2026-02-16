"""
Legal Theme Extractor

Extract high-level themes from clusters of related entities in the knowledge graph.
Based on LightRAG's high-level retrieval concept.

Features:
- Graph partitioning into connected subgraphs
- LLM-based theme extraction from entity clusters
- Theme index for semantic search
- FAISS-based theme embedding index

Usage:
    >>> from vn_legal_rag.offline.theme_extractor import ThemeExtractor
    >>> extractor = ThemeExtractor(provider="gemini", model="gemini-2.0-flash")
    >>> themes = extractor.extract_themes(kg)
    >>> index = extractor.build_index(themes)
"""

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..utils.simple_logger import get_logger

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


@dataclass
class Theme:
    """High-level theme extracted from entity cluster."""

    id: str
    name: str  # Vietnamese theme name
    description: str  # 1-2 sentence description
    source_entities: List[str] = field(default_factory=list)  # Entity IDs
    source_relations: List[str] = field(default_factory=list)  # Relation types
    keywords: List[str] = field(default_factory=list)  # Search keywords
    embedding: Optional[np.ndarray] = None  # For vector search
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.embedding is not None:
            d["embedding"] = self.embedding.tolist()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Theme":
        if "embedding" in data and data["embedding"]:
            data["embedding"] = np.array(data["embedding"])
        return cls(**data)


@dataclass
class Subgraph:
    """A connected subgraph of entities and relations."""

    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]

    @property
    def entity_ids(self) -> Set[str]:
        return {e.get("id", "") for e in self.entities}

    @property
    def relation_types(self) -> Set[str]:
        return {r.get("type", "") for r in self.relations}


THEME_EXTRACTION_PROMPT_VI = """Bạn là chuyên gia pháp lý Việt Nam. Phân tích nhóm entities sau và xác định chủ đề pháp lý:

**Entities trong nhóm:**
{entities}

**Quan hệ giữa các entities:**
{relations}

Hãy xác định 1-3 chủ đề pháp lý cấp cao cho nhóm này. Mỗi chủ đề:
- Tên ngắn gọn (2-5 từ tiếng Việt)
- Mô tả 1 câu
- 3-5 từ khóa tìm kiếm

Trả về JSON:
{{
  "themes": [
    {{
      "name": "Tên chủ đề",
      "description": "Mô tả ngắn",
      "keywords": ["từ khóa 1", "từ khóa 2", ...]
    }}
  ]
}}

CHỈ trả về JSON."""


class ThemeExtractor:
    """Extract high-level themes from knowledge graph."""

    def __init__(
        self,
        provider: str = "gemini",
        model: str = "gemini-2.0-flash",
        min_cluster_size: int = 3,
        max_cluster_size: int = 50,
        embedding_dim: int = 384,
        use_cache: bool = True,
        cache_db_path: str = "data/llm_cache.db",
    ):
        """
        Initialize theme extractor.

        Args:
            provider: LLM provider
            model: Model name
            min_cluster_size: Minimum entities per cluster for theme extraction
            max_cluster_size: Maximum entities (sample if larger)
            embedding_dim: Embedding dimension for themes
            use_cache: Whether to cache LLM responses
            cache_db_path: Path to SQLite cache database
        """
        self.provider_name = provider
        self.model = model
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.embedding_dim = embedding_dim
        self._use_cache = use_cache
        self._cache_db_path = cache_db_path

        self.logger = get_logger("theme_extractor")
        self._llm = None
        self._embedder = None

    def _get_llm(self):
        """Lazy load LLM with optional caching."""
        if self._llm is None:
            from ..utils.basic_llm_provider import create_llm_provider

            self._llm = create_llm_provider(
                self.provider_name,
                model=self.model,
                use_cache=self._use_cache,
                cache_db_path=self._cache_db_path,
            )
        return self._llm

    def _get_embedder(self):
        """Lazy load embedder."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
            except ImportError:
                self.logger.warning("sentence-transformers not available")
                self._embedder = False
        return self._embedder if self._embedder else None

    def partition_graph(self, kg: Dict[str, Any]) -> List[Subgraph]:
        """
        Partition KG into connected subgraphs using Union-Find.

        Args:
            kg: Knowledge graph dict with 'entities' and 'relationships'

        Returns:
            List of Subgraph objects
        """
        entities = kg.get("entities", [])
        relations = kg.get("relationships", [])

        if not entities:
            return []

        # Build entity lookup
        entity_map = {e.get("id", ""): e for e in entities if e.get("id")}
        entity_ids = list(entity_map.keys())

        # Union-Find for connected components
        parent = {eid: eid for eid in entity_ids}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union entities connected by relations
        for rel in relations:
            source = rel.get("source_id", rel.get("source", ""))
            target = rel.get("target_id", rel.get("target", ""))
            if source in parent and target in parent:
                union(source, target)

        # Group entities by component
        components = defaultdict(list)
        for eid in entity_ids:
            components[find(eid)].append(eid)

        # Build subgraphs
        subgraphs = []
        for component_ids in components.values():
            if len(component_ids) < self.min_cluster_size:
                continue

            component_set = set(component_ids)

            # Sample if too large
            if len(component_ids) > self.max_cluster_size:
                import random

                component_ids = random.sample(component_ids, self.max_cluster_size)
                component_set = set(component_ids)

            # Get entities
            subgraph_entities = [entity_map[eid] for eid in component_ids]

            # Get relations within component
            subgraph_relations = [
                r
                for r in relations
                if r.get("source_id", r.get("source", "")) in component_set
                and r.get("target_id", r.get("target", "")) in component_set
            ]

            subgraphs.append(
                Subgraph(
                    entities=subgraph_entities,
                    relations=subgraph_relations,
                )
            )

        self.logger.info(f"Partitioned into {len(subgraphs)} subgraphs")
        return subgraphs

    def extract_themes_from_subgraph(self, subgraph: Subgraph) -> List[Theme]:
        """
        Extract themes from a single subgraph via LLM.

        Args:
            subgraph: Subgraph with entities and relations

        Returns:
            List of Theme objects
        """
        # Format entities
        entity_lines = []
        for e in subgraph.entities[:20]:  # Limit for prompt length
            name = e.get("name", e.get("text", ""))
            etype = e.get("type", e.get("label", ""))
            entity_lines.append(f"- {name} [{etype}]")

        # Format relations
        relation_lines = []
        seen = set()
        for r in subgraph.relations[:15]:
            rtype = r.get("type", "")
            if rtype not in seen:
                relation_lines.append(f"- {rtype}")
                seen.add(rtype)

        if not entity_lines:
            return []

        prompt = THEME_EXTRACTION_PROMPT_VI.format(
            entities="\n".join(entity_lines),
            relations="\n".join(relation_lines) if relation_lines else "(không có)",
        )

        try:
            llm = self._get_llm()
            response = llm.generate(prompt)
            return self._parse_themes_response(response, subgraph)
        except Exception as e:
            self.logger.warning(f"Theme extraction failed: {e}")
            return self._fallback_themes(subgraph)

    def _parse_themes_response(
        self,
        response: str,
        subgraph: Subgraph,
    ) -> List[Theme]:
        """Parse LLM response to Theme objects."""
        try:
            json_match = re.search(r"\{[\s\S]*\}", response)
            if not json_match:
                return self._fallback_themes(subgraph)

            data = json.loads(json_match.group())
            themes_data = data.get("themes", [])

            themes = []
            for idx, td in enumerate(themes_data):
                theme_id = hashlib.sha256(
                    f"{td.get('name', '')}:{idx}".encode()
                ).hexdigest()[:16]

                themes.append(
                    Theme(
                        id=theme_id,
                        name=td.get("name", ""),
                        description=td.get("description", ""),
                        source_entities=list(subgraph.entity_ids)[:20],
                        source_relations=list(subgraph.relation_types),
                        keywords=td.get("keywords", []),
                        confidence=0.8,
                    )
                )

            return themes
        except json.JSONDecodeError:
            return self._fallback_themes(subgraph)

    def _fallback_themes(self, subgraph: Subgraph) -> List[Theme]:
        """Create fallback theme from entity types."""
        type_counts = defaultdict(int)
        for e in subgraph.entities:
            etype = e.get("type", e.get("label", "UNKNOWN"))
            type_counts[etype] += 1

        if not type_counts:
            return []

        top_type = max(type_counts, key=type_counts.get)
        theme_id = hashlib.sha256(f"fallback:{top_type}".encode()).hexdigest()[:16]

        return [
            Theme(
                id=theme_id,
                name=f"Nhóm {top_type}",
                description=f"Entities loại {top_type}",
                source_entities=list(subgraph.entity_ids)[:20],
                source_relations=list(subgraph.relation_types),
                keywords=[top_type.lower()],
                confidence=0.5,
            )
        ]

    def extract_themes(
        self,
        kg: Dict[str, Any],
        show_progress: bool = True,
    ) -> List[Theme]:
        """
        Extract themes from entire knowledge graph.

        Args:
            kg: Knowledge graph dict
            show_progress: Print progress

        Returns:
            List of Theme objects
        """
        subgraphs = self.partition_graph(kg)

        all_themes = []
        for idx, sg in enumerate(subgraphs):
            if show_progress:
                print(f"  [Theme] {idx + 1}/{len(subgraphs)}: {len(sg.entities)} entities")

            themes = self.extract_themes_from_subgraph(sg)
            all_themes.extend(themes)

        self.logger.info(
            f"Extracted {len(all_themes)} themes from {len(subgraphs)} subgraphs"
        )
        return all_themes

    def add_embeddings(self, themes: List[Theme]) -> List[Theme]:
        """Add embeddings to themes for vector search."""
        embedder = self._get_embedder()
        if not embedder:
            return themes

        for theme in themes:
            text = f"{theme.name}. {theme.description}. {' '.join(theme.keywords)}"
            try:
                theme.embedding = embedder.encode(text)
            except Exception as e:
                self.logger.warning(f"Embedding failed for theme {theme.id}: {e}")

        return themes

    def build_index(self, themes: List[Theme]) -> Optional["ThemeIndex"]:
        """Build FAISS index from themes."""
        if not FAISS_AVAILABLE:
            self.logger.warning("FAISS not available, skipping index build")
            return None

        themes = self.add_embeddings(themes)
        return ThemeIndex(themes, self.embedding_dim)


class ThemeIndex:
    """FAISS-based index for theme search."""

    def __init__(self, themes: List[Theme], embedding_dim: int = 384):
        """
        Initialize theme index.

        Args:
            themes: List of Theme objects with embeddings
            embedding_dim: Embedding dimension
        """
        self.themes = themes
        self.embedding_dim = embedding_dim
        self.index = None

        self._build_index()

    def _build_index(self):
        """Build FAISS index."""
        if not FAISS_AVAILABLE:
            return

        valid_themes = [t for t in self.themes if t.embedding is not None]
        if not valid_themes:
            return

        embeddings = np.vstack([t.embedding for t in valid_themes]).astype(np.float32)
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        self.themes = valid_themes

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[Theme, float]]:
        """
        Search for similar themes.

        Args:
            query_embedding: Query embedding vector
            k: Number of results

        Returns:
            List of (Theme, score) tuples
        """
        if self.index is None or not self.themes:
            return []

        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, min(k, len(self.themes)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.themes):
                results.append((self.themes[idx], float(score)))

        return results

    def save(self, path: str):
        """Save index to disk."""
        if self.index is None:
            return

        from pathlib import Path

        p = Path(path)
        faiss.write_index(self.index, str(p.with_suffix(".faiss")))

        themes_data = [t.to_dict() for t in self.themes]
        with open(p.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
            json.dump(themes_data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> Optional["ThemeIndex"]:
        """Load index from disk."""
        if not FAISS_AVAILABLE:
            return None

        from pathlib import Path

        p = Path(path)
        faiss_path = p.with_suffix(".faiss")
        meta_path = p.with_suffix(".meta.json")

        if not faiss_path.exists() or not meta_path.exists():
            return None

        with open(meta_path, "r", encoding="utf-8") as f:
            themes_data = json.load(f)
        themes = [Theme.from_dict(td) for td in themes_data]

        instance = cls.__new__(cls)
        instance.themes = themes
        instance.embedding_dim = (
            themes[0].embedding.shape[0]
            if themes and themes[0].embedding is not None
            else 384
        )
        instance.index = faiss.read_index(str(faiss_path))

        return instance
