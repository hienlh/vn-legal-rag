"""
Data loaders for KG, forest, and summaries.

Provides utility functions to load data files with error handling.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def load_kg(kg_path: str) -> Dict[str, Any]:
    """
    Load knowledge graph from JSON file.

    Args:
        kg_path: Path to KG JSON file

    Returns:
        KG dictionary with entities and relations

    Raises:
        FileNotFoundError: If KG file doesn't exist
        json.JSONDecodeError: If KG file is invalid JSON
    """
    kg_file = Path(kg_path)

    if not kg_file.exists():
        raise FileNotFoundError(f"KG file not found: {kg_path}")

    logger.info(f"Loading KG from: {kg_path}")

    with open(kg_file, "r", encoding="utf-8") as f:
        kg = json.load(f)

    # Validate KG structure
    if "entities" not in kg:
        raise ValueError(f"Invalid KG format: missing 'entities' key")

    # Normalize: accept both 'relations' and 'relationships' keys
    if "relations" not in kg:
        if "relationships" in kg:
            kg["relations"] = kg["relationships"]
        else:
            kg["relations"] = []  # Default to empty list if missing

    logger.info(
        f"Loaded KG: {len(kg.get('entities', []))} entities, "
        f"{len(kg.get('relations', []))} relations"
    )

    return kg


def load_forest(forest_path: str):
    """
    Load unified forest from JSON file.

    Args:
        forest_path: Path to forest JSON file

    Returns:
        UnifiedForest instance

    Raises:
        FileNotFoundError: If forest file doesn't exist
    """
    from vn_legal_rag.types import UnifiedForest

    forest_file = Path(forest_path)

    if not forest_file.exists():
        raise FileNotFoundError(f"Forest file not found: {forest_path}")

    logger.info(f"Loading forest from: {forest_path}")

    with open(forest_file, "r", encoding="utf-8") as f:
        forest_data = f.read()

    forest = UnifiedForest.from_json(forest_data)

    logger.info(f"Loaded forest: {len(forest.trees)} documents")

    return forest


def load_summaries(summaries_path: str) -> Optional[Dict[str, Any]]:
    """
    Load chapter or article summaries from JSON file.

    Args:
        summaries_path: Path to summaries JSON file

    Returns:
        Summaries dictionary or None if file doesn't exist
    """
    summaries_file = Path(summaries_path)

    if not summaries_file.exists():
        logger.warning(f"Summaries file not found: {summaries_path}")
        return None

    logger.info(f"Loading summaries from: {summaries_path}")

    with open(summaries_file, "r", encoding="utf-8") as f:
        summaries = json.load(f)

    logger.info(f"Loaded {len(summaries)} summaries")

    return summaries


def load_training_data(csv_path: str) -> list:
    """
    Load training/test data from CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of training examples

    Raises:
        FileNotFoundError: If CSV file doesn't exist
    """
    import csv

    csv_file = Path(csv_path)

    if not csv_file.exists():
        raise FileNotFoundError(f"Training data file not found: {csv_path}")

    logger.info(f"Loading training data from: {csv_path}")

    examples = []

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            examples.append(row)

    logger.info(f"Loaded {len(examples)} training examples")

    return examples


def save_json(
    data: Any,
    output_path: str,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """
    Save data to JSON file with atomic write.

    Args:
        data: Data to save
        output_path: Output file path
        indent: JSON indentation
        ensure_ascii: Whether to escape non-ASCII characters
    """
    import tempfile

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: write to temp file then rename
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=output_file.parent,
        suffix=".tmp"
    ) as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)
        temp_path = f.name

    Path(temp_path).rename(output_file)

    logger.info(f"Saved JSON to: {output_path}")


def load_json(json_path: str) -> Any:
    """
    Load data from JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        Loaded data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is invalid JSON
    """
    json_file = Path(json_path)

    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def build_forest_from_db(
    db_path: str,
    chapter_summaries: Optional[Dict] = None,
    doc_ids: Optional[list] = None,
):
    """
    Build UnifiedForest from database.

    Args:
        db_path: Path to SQLite database
        chapter_summaries: Optional chapter summaries for descriptions
        doc_ids: Optional list of document IDs to include (default: all)

    Returns:
        UnifiedForest instance
    """
    import sqlite3
    from vn_legal_rag.types import TreeNode, TreeIndex, UnifiedForest, NodeType

    if chapter_summaries is None:
        chapter_summaries = {}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get document IDs
    if doc_ids is None:
        cursor = conn.execute("SELECT id FROM legal_documents")
        doc_ids = [row["id"] for row in cursor]

    logger.info(f"Building forest for {len(doc_ids)} document(s)...")

    forest = UnifiedForest()

    for doc_id in doc_ids:
        # Get document
        doc = conn.execute(
            "SELECT * FROM legal_documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if not doc:
            continue

        # Get chapters
        chapters = conn.execute(
            "SELECT * FROM legal_chapters WHERE document_id = ? ORDER BY position",
            (doc_id,)
        ).fetchall()

        sub_nodes = []
        for chapter in chapters:
            chapter_children = []

            # Get articles directly under chapter
            articles = conn.execute(
                "SELECT * FROM legal_articles WHERE chapter_id = ? AND section_id IS NULL ORDER BY position",
                (chapter["id"],)
            ).fetchall()

            for article in articles:
                article_node = TreeNode(
                    node_id=article["id"],
                    node_type=NodeType.ARTICLE,
                    name=article["title"] or f"Điều {article['article_number']}",
                    description="",
                    content=(article["raw_text"] or "")[:500],
                    metadata={"article_number": article["article_number"]},
                    sub_nodes=[],
                )
                chapter_children.append(article_node)

            # Get sections in chapter
            sections = conn.execute(
                "SELECT * FROM legal_sections WHERE chapter_id = ? ORDER BY position",
                (chapter["id"],)
            ).fetchall()

            for section in sections:
                section_articles = conn.execute(
                    "SELECT * FROM legal_articles WHERE section_id = ? ORDER BY position",
                    (section["id"],)
                ).fetchall()

                for article in section_articles:
                    article_node = TreeNode(
                        node_id=article["id"],
                        node_type=NodeType.ARTICLE,
                        name=article["title"] or f"Điều {article['article_number']}",
                        description="",
                        content=(article["raw_text"] or "")[:500],
                        metadata={"article_number": article["article_number"]},
                        sub_nodes=[],
                    )
                    chapter_children.append(article_node)

            if chapter_children:
                # Get chapter description from summaries
                description = ""
                if isinstance(chapter_summaries, dict):
                    if "summaries" in chapter_summaries:
                        # Handle {"summaries": [...]} format
                        for s in chapter_summaries.get("summaries", []):
                            if s.get("chapter_id") == chapter["id"]:
                                description = s.get("description", "")
                                break
                    else:
                        # Handle {chapter_id: {...}} format
                        chapter_data = chapter_summaries.get(chapter["id"], {})
                        if isinstance(chapter_data, dict):
                            description = chapter_data.get("description", "")
                        else:
                            description = str(chapter_data)

                chapter_node = TreeNode(
                    node_id=chapter["id"],
                    node_type=NodeType.CHAPTER,
                    name=chapter["title"] or f"Chương {chapter['chapter_number']}",
                    description=description,
                    content="",
                    metadata={"chapter_number": chapter["chapter_number"]},
                    sub_nodes=chapter_children,
                )
                sub_nodes.append(chapter_node)

        if sub_nodes:
            root = TreeNode(
                node_id=doc["id"],
                node_type=NodeType.DOCUMENT,
                name=doc["title"],
                description=f"{doc['loai_van_ban']} số {doc['so_hieu']}",
                content="",
                metadata={"so_hieu": doc["so_hieu"]},
                sub_nodes=sub_nodes,
            )
            tree = TreeIndex(root=root, doc_id=doc["id"])
            forest.add_tree(tree)

    conn.close()

    logger.info(f"Forest built: {len(forest.trees)} trees")
    return forest


__all__ = [
    "load_kg",
    "load_forest",
    "load_summaries",
    "load_training_data",
    "save_json",
    "load_json",
    "build_forest_from_db",
]
