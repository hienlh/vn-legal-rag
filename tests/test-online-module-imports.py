"""
Test Online Module Imports

Verifies that all online phase modules can be imported correctly
despite using kebab-case filenames.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_main_imports():
    """Test main entry point imports."""
    print("Testing main imports...")
    from vn_legal_rag.online import (
        LegalGraphRAG,
        GraphRAGResponse,
        create_legal_graphrag,
    )
    print("✓ Main imports successful")
    return True


def test_tree_retriever_imports():
    """Test tree traversal retriever imports."""
    print("Testing tree retriever imports...")
    from vn_legal_rag.online import (
        TreeTraversalRetriever,
        TreeSearchResult,
        build_tree_retriever,
    )
    print("✓ Tree retriever imports successful")
    return True


def test_dual_retriever_imports():
    """Test dual-level retriever imports."""
    print("Testing dual retriever imports...")
    from vn_legal_rag.online import (
        DualLevelRetriever,
        DualLevelResult,
        DualLevelConfig,
        LowLevelResult,
        HighLevelResult,
    )
    print("✓ Dual retriever imports successful")
    return True


def test_semantic_bridge_imports():
    """Test semantic bridge imports."""
    print("Testing semantic bridge imports...")
    from vn_legal_rag.online import (
        SemanticBridge,
        create_semantic_bridge,
    )
    print("✓ Semantic bridge imports successful")
    return True


def test_query_analyzer_imports():
    """Test query analyzer imports."""
    print("Testing query analyzer imports...")
    from vn_legal_rag.online import (
        VietnameseLegalQueryAnalyzer,
        AnalyzedQuery,
        ExpandedQuery,
        QueryIntent,
        LegalQueryType,
        QueryTypeConfig,
        expand_query,
        analyze_query,
    )
    print("✓ Query analyzer imports successful")
    return True


def test_ppr_imports():
    """Test PPR imports."""
    print("Testing PPR imports...")
    from vn_legal_rag.online import (
        PersonalizedPageRank,
        PPRResult,
        PPRConfig,
    )
    print("✓ PPR imports successful")
    return True


def test_query_expansion():
    """Test query expansion functionality."""
    print("\nTesting query expansion...")
    from vn_legal_rag.online import expand_query

    query = "CTCP có cần ĐKKD không?"
    expanded = expand_query(query)

    print(f"Original: {expanded.original}")
    print(f"Expanded: {expanded.expanded}")
    print(f"Abbreviations: {expanded.abbreviations_found}")
    print("✓ Query expansion working")
    return True


def test_query_analysis():
    """Test full query analysis."""
    print("\nTesting query analysis...")
    from vn_legal_rag.online import analyze_query

    query = "Điều 5 quy định gì về thành lập công ty?"
    analyzed = analyze_query(query)

    print(f"Query type: {analyzed.query_type.value}")
    print(f"Intent: {analyzed.intent.value}")
    print(f"Article refs: {analyzed.article_refs}")
    print(f"Keywords: {analyzed.keywords[:5]}")

    # Verify basic functionality
    assert analyzed.query_type is not None, "Query type should not be None"
    assert analyzed.intent is not None, "Intent should not be None"
    assert len(analyzed.article_refs) > 0, "Should detect article reference"

    print("✓ Query analysis working")
    return True


def main():
    """Run all import tests."""
    print("=" * 60)
    print("Online Module Import Tests")
    print("=" * 60)

    tests = [
        test_main_imports,
        test_tree_retriever_imports,
        test_dual_retriever_imports,
        test_semantic_bridge_imports,
        test_query_analyzer_imports,
        test_ppr_imports,
        test_query_expansion,
        test_query_analysis,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            import traceback
            print(f"✗ {test.__name__} failed: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
