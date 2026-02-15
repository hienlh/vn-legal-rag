"""
Ablation configuration for RAG component testing.

Focus on components that differentiate from PageIndex and LightRAG:
- PageIndex contribution: Tree Traversal (Tier 1)
- LightRAG contribution: Keyphrase matching in DualLevel (Tier 2)
- Our contributions:
  - Semantic Bridge (agreement-based merge between Tree & DualLevel)
  - KG Expansion (cross-chapter relations via KG)
  - RRF Merge (intelligent rank fusion)
  - Adjacent Expansion (nearby article recovery)

Usage:
    >>> from vn_legal_rag.types import AblationConfig, get_paper_ablation_configs
    >>> configs = get_paper_ablation_configs()
    >>> rag = LegalGraphRAG(..., ablation_config=configs["no_semantic_bridge"])
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class AblationConfig:
    """Configuration for ablation studies."""

    # === Tier 1: Tree Traversal (PageIndex-style) ===
    enable_tree: bool = True

    # === Tier 2: DualLevel Retriever (LightRAG-style keyphrase) ===
    enable_dual_level: bool = True
    enable_dual_low_level: bool = True
    enable_dual_high_level: bool = True

    # === Tier 2 Sub-components ===
    enable_ppr: bool = True
    enable_keyphrase: bool = True
    enable_semantic: bool = True
    enable_concept: bool = True
    enable_theme: bool = True
    enable_hierarchy: bool = True

    # === Tier 3: Our Contributions ===
    enable_intelligent_merge: bool = True  # RRF merge vs simple concat
    enable_kg_expansion: bool = True       # Cross-chapter KG relations
    enable_semantic_bridge: bool = True    # Agreement-based score fusion
    enable_adjacent_expansion: bool = True # Adjacent article expansion

    # === Stage 2: Cross-Encoder Reranking ===
    enable_reranker: bool = True  # Cross-encoder reranking (TOP 100 -> TOP 10)

    # === Query Processing ===
    enable_query_type_classification: bool = True  # Hybrid rules + LLM classification
    enable_ontology_expansion: bool = True         # Ontology-based query expansion
    enable_document_filter: bool = True            # Document-aware result filtering

    # === Score Calibration ===
    enable_adaptive_threshold: bool = True         # Tree-DualLevel agreement threshold
    enable_ambiguity_calibration: bool = True      # Query ambiguity calibration

    # === Other ===
    enable_kg_retrieval: bool = True

    # Config name for reporting
    name: str = "full"

    def to_dict(self) -> Dict[str, bool]:
        """Convert to dict for serialization."""
        return {
            "name": self.name,
            "enable_tree": self.enable_tree,
            "enable_dual_level": self.enable_dual_level,
            "enable_ppr": self.enable_ppr,
            "enable_intelligent_merge": self.enable_intelligent_merge,
            "enable_kg_expansion": self.enable_kg_expansion,
            "enable_semantic_bridge": self.enable_semantic_bridge,
            "enable_adjacent_expansion": self.enable_adjacent_expansion,
            "enable_reranker": self.enable_reranker,
            "enable_query_type_classification": self.enable_query_type_classification,
            "enable_ontology_expansion": self.enable_ontology_expansion,
            "enable_document_filter": self.enable_document_filter,
            "enable_adaptive_threshold": self.enable_adaptive_threshold,
            "enable_ambiguity_calibration": self.enable_ambiguity_calibration,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "AblationConfig":
        """Create from dict."""
        return cls(**{k: v for k, v in d.items() if k != "name"}, name=d.get("name", "custom"))


def get_ablation_configs() -> Dict[str, AblationConfig]:
    """Get all ablation configurations."""
    return get_paper_ablation_configs()


def get_paper_ablation_configs() -> Dict[str, AblationConfig]:
    """
    Ablation configs for paper - focus on our contributions vs baselines.

    Comparison framework:
    - full: Our complete 3-Tier system
    - tree_only: PageIndex baseline (Tier 1 only)
    - dual_only: LightRAG-style baseline (Tier 2 keyphrase only)
    - no_X: Remove one component to measure its contribution
    """
    return {
        # ========== FULL SYSTEM ==========
        "full": AblationConfig(name="full"),

        # ========== BASELINES (for comparison) ==========
        # PageIndex baseline: Tree navigation only, no fusion
        "tree_only": AblationConfig(
            enable_dual_level=False,
            enable_kg_expansion=False,
            enable_semantic_bridge=False,
            enable_adjacent_expansion=False,
            enable_intelligent_merge=False,
            name="tree_only",
        ),

        # LightRAG-style baseline: DualLevel keyphrase only (no tree)
        "dual_only": AblationConfig(
            enable_tree=False,
            enable_kg_expansion=False,
            enable_semantic_bridge=False,
            enable_adjacent_expansion=False,
            enable_intelligent_merge=False,
            name="dual_only",
        ),

        # DualLevel without PPR (to measure PPR contribution in isolation)
        "dual_no_ppr": AblationConfig(
            enable_tree=False,
            enable_ppr=False,
            enable_kg_expansion=False,
            enable_semantic_bridge=False,
            enable_adjacent_expansion=False,
            enable_intelligent_merge=False,
            name="dual_no_ppr",
        ),

        # ========== OUR CONTRIBUTIONS (remove one at a time) ==========
        # Without PPR (Personalized PageRank in DualLevel)
        "no_ppr": AblationConfig(
            enable_ppr=False,
            name="no_ppr",
        ),

        # Without Semantic Bridge (agreement-based merge)
        "no_semantic_bridge": AblationConfig(
            enable_semantic_bridge=False,
            name="no_semantic_bridge",
        ),

        # Without KG Expansion (cross-chapter relations)
        "no_kg_expansion": AblationConfig(
            enable_kg_expansion=False,
            name="no_kg_expansion",
        ),

        # Without RRF Merge (use simple concatenation)
        "no_rrf_merge": AblationConfig(
            enable_intelligent_merge=False,
            name="no_rrf_merge",
        ),

        # Without Adjacent Expansion
        "no_adjacent": AblationConfig(
            enable_adjacent_expansion=False,
            name="no_adjacent",
        ),

        # ========== TIER ISOLATION ==========
        # Without Tree (Tier 1) - test Tier 2 contribution
        "no_tree": AblationConfig(
            enable_tree=False,
            name="no_tree",
        ),

        # Without DualLevel (Tier 2) - test Tier 1 contribution
        "no_dual": AblationConfig(
            enable_dual_level=False,
            name="no_dual",
        ),

        # ========== COMBINED ABLATIONS ==========
        # Simple merge (no RRF, no bridge, no adjacent, no kg_exp)
        # This shows the value of ALL our Tier 3 contributions
        "simple_merge": AblationConfig(
            enable_intelligent_merge=False,
            enable_semantic_bridge=False,
            enable_adjacent_expansion=False,
            enable_kg_expansion=False,
            name="simple_merge",
        ),

        # ========== RERANKER ABLATION ==========
        # Without cross-encoder reranker (measure reranker contribution)
        "no_reranker": AblationConfig(
            enable_reranker=False,
            name="no_reranker",
        ),

        # Without Tree AND Reranker (KG only, baseline speed)
        "no_tree_no_reranker": AblationConfig(
            enable_tree=False,
            enable_reranker=False,
            name="no_tree_no_reranker",
        ),

        # Without KG (Tree only) - test Tree-only contribution
        "no_kg": AblationConfig(
            enable_dual_level=False,
            enable_kg_retrieval=False,
            enable_kg_expansion=False,
            name="no_kg",
        ),

        # ========== QUERY PROCESSING ABLATIONS ==========
        # Without Query Type Classification
        "no_query_type": AblationConfig(
            enable_query_type_classification=False,
            name="no_query_type",
        ),

        # Without Ontology Expansion
        "no_ontology": AblationConfig(
            enable_ontology_expansion=False,
            name="no_ontology",
        ),

        # Without Document Filter
        "no_doc_filter": AblationConfig(
            enable_document_filter=False,
            name="no_doc_filter",
        ),

        # ========== CALIBRATION ABLATIONS ==========
        # Without Adaptive Threshold
        "no_adaptive_threshold": AblationConfig(
            enable_adaptive_threshold=False,
            name="no_adaptive_threshold",
        ),

        # Without Ambiguity Calibration
        "no_ambiguity_calibration": AblationConfig(
            enable_ambiguity_calibration=False,
            name="no_ambiguity_calibration",
        ),

        # Without both calibrations
        "no_calibration": AblationConfig(
            enable_adaptive_threshold=False,
            enable_ambiguity_calibration=False,
            name="no_calibration",
        ),
    }
