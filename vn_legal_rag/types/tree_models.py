"""
Tree node models for hierarchical document indexing.

Vietnamese legal doc hierarchy: Document→Chương→Mục→Điều→Khoản→Điểm
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import json


class NodeType(str, Enum):
    """Node types in legal document hierarchy."""
    DOCUMENT = "document"
    CHAPTER = "chapter"        # Chương
    SECTION = "section"        # Mục
    ARTICLE = "article"        # Điều
    CLAUSE = "clause"          # Khoản
    POINT = "point"            # Điểm


@dataclass
class TreeNode:
    """
    Hierarchical node for legal document tree.
    """
    node_id: str
    node_type: NodeType
    name: str
    description: str = ""
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    sub_nodes: List["TreeNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "name": self.name,
            "description": self.description,
            "content": self.content,
            "metadata": self.metadata,
            "sub_nodes": [node.to_dict() for node in self.sub_nodes],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TreeNode":
        """Deserialize from dictionary."""
        sub_nodes = [cls.from_dict(n) for n in data.get("sub_nodes", [])]
        return cls(
            node_id=data["node_id"],
            node_type=NodeType(data["node_type"]),
            name=data["name"],
            description=data.get("description", ""),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            sub_nodes=sub_nodes,
        )

    def find_node(self, node_id: str) -> Optional["TreeNode"]:
        """Find node by ID in subtree."""
        if self.node_id == node_id:
            return self
        for child in self.sub_nodes:
            result = child.find_node(node_id)
            if result:
                return result
        return None


@dataclass
class CrossRefEdge:
    """Cross-reference edge between nodes (possibly across documents)."""
    source_node_id: str
    target_node_id: str
    reference_text: str
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "reference_text": self.reference_text,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrossRefEdge":
        """Deserialize from dictionary."""
        return cls(
            source_node_id=data["source_node_id"],
            target_node_id=data["target_node_id"],
            reference_text=data["reference_text"],
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class ForestStats:
    """Statistics for unified forest."""
    doc_count: int
    total_nodes: int
    cross_ref_count: int
    node_type_counts: Dict[NodeType, int] = field(default_factory=dict)


class TreeIndex:
    """
    Tree index for a single legal document.
    """

    def __init__(self, root: TreeNode, doc_id: str):
        """
        Initialize tree index.

        Args:
            root: Root TreeNode (document level)
            doc_id: Document ID (e.g., "59-2020-QH14")
        """
        self.root = root
        self.doc_id = doc_id
        self._node_cache: Optional[Dict[str, TreeNode]] = None

    def _build_node_cache(self) -> Dict[str, TreeNode]:
        """Build flat index of all nodes for fast lookup."""
        cache = {}

        def _traverse(node: TreeNode):
            cache[node.node_id] = node
            for child in node.sub_nodes:
                _traverse(child)

        _traverse(self.root)
        return cache

    def find_node(self, node_id: str) -> Optional[TreeNode]:
        """Find node by ID (cached)."""
        if self._node_cache is None:
            self._node_cache = self._build_node_cache()
        return self._node_cache.get(node_id)

    def get_path(self, node_id: str) -> List[TreeNode]:
        """
        Get path from root to node.

        Returns:
            List of nodes from root to target (empty if not found)
        """
        path = []

        def _find_path(node: TreeNode, target_id: str) -> bool:
            path.append(node)
            if node.node_id == target_id:
                return True
            for child in node.sub_nodes:
                if _find_path(child, target_id):
                    return True
            path.pop()
            return False

        _find_path(self.root, node_id)
        return path

    def to_json(self) -> str:
        """Export to JSON string."""
        data = {
            "doc_id": self.doc_id,
            "root": self.root.to_dict(),
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "TreeIndex":
        """Load from JSON string."""
        data = json.loads(json_str)
        root = TreeNode.from_dict(data["root"])
        return cls(root=root, doc_id=data["doc_id"])


class UnifiedForest:
    """
    Unified forest of legal document trees with cross-document links.
    """

    def __init__(self):
        """Initialize empty forest."""
        self.trees: Dict[str, TreeIndex] = {}
        self.cross_refs: List[CrossRefEdge] = []
        self._global_node_index: Optional[Dict[str, TreeNode]] = None

    def add_tree(self, tree: TreeIndex):
        """Add document tree to forest."""
        self.trees[tree.doc_id] = tree
        self._global_node_index = None  # Invalidate cache

    def get_tree(self, doc_id: str) -> Optional[TreeIndex]:
        """Get tree by document ID."""
        return self.trees.get(doc_id)

    def add_cross_ref(self, edge: CrossRefEdge):
        """Add cross-reference edge."""
        self.cross_refs.append(edge)

    def _build_global_index(self) -> Dict[str, TreeNode]:
        """Build global node index across all trees."""
        index = {}
        for tree in self.trees.values():
            if tree._node_cache is None:
                tree._node_cache = tree._build_node_cache()
            index.update(tree._node_cache)
        return index

    def find_node(self, node_id: str) -> Optional[TreeNode]:
        """Find node by ID across all trees."""
        if self._global_node_index is None:
            self._global_node_index = self._build_global_index()
        return self._global_node_index.get(node_id)

    def get_related_nodes(self, node_id: str) -> List[TreeNode]:
        """Get nodes referenced by this node via cross-refs."""
        related = []
        for edge in self.cross_refs:
            if edge.source_node_id == node_id:
                target = self.find_node(edge.target_node_id)
                if target:
                    related.append(target)
        return related

    def get_path_to_root(self, node_id: str) -> List[TreeNode]:
        """
        Get path from root to node (alias for compatibility).

        Searches all trees to find the node.

        Returns:
            List of nodes from root to target (empty if not found)
        """
        for tree in self.trees.values():
            path = tree.get_path(node_id)
            if path:
                return path
        return []

    def stats(self) -> ForestStats:
        """Compute forest statistics."""
        total_nodes = 0
        node_type_counts: Dict[NodeType, int] = {}

        for tree in self.trees.values():
            if tree._node_cache is None:
                tree._node_cache = tree._build_node_cache()

            total_nodes += len(tree._node_cache)

            for node in tree._node_cache.values():
                node_type_counts[node.node_type] = (
                    node_type_counts.get(node.node_type, 0) + 1
                )

        return ForestStats(
            doc_count=len(self.trees),
            total_nodes=total_nodes,
            cross_ref_count=len(self.cross_refs),
            node_type_counts=node_type_counts,
        )

    def to_json(self) -> str:
        """Export forest to JSON."""
        data = {
            "trees": {
                doc_id: json.loads(tree.to_json())
                for doc_id, tree in self.trees.items()
            },
            "cross_refs": [edge.to_dict() for edge in self.cross_refs],
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "UnifiedForest":
        """Load forest from JSON."""
        data = json.loads(json_str)
        forest = cls()

        # Load trees
        for doc_id, tree_data in data.get("trees", {}).items():
            tree_json = json.dumps(tree_data)
            tree = TreeIndex.from_json(tree_json)
            forest.add_tree(tree)

        # Load cross-refs
        for edge_data in data.get("cross_refs", []):
            edge = CrossRefEdge.from_dict(edge_data)
            forest.add_cross_ref(edge)

        return forest
