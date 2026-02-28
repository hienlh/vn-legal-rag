"""
Class Hierarchy Builder for Legal Ontology.

Infers class hierarchy from KG entity types and maps to base ontology.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class HierarchyMapping:
    """Mapping from entity type to ontology class with parent."""
    entity_type: str
    class_name: str
    parent_class: str
    vietnamese_label: str
    confidence: float = 1.0


class ClassHierarchyBuilder:
    """
    Build class hierarchy from KG entity types.

    Maps entity types to base ontology classes and infers parent-child
    relationships based on naming patterns and type semantics.
    """

    # Pattern-based mappings for Vietnamese legal domain
    TYPE_PATTERNS = {
        # Organization patterns
        r"(?i)(công\s*ty|company|corp|enterprise|doanh\s*nghiệp)": ("Enterprise", "Organization"),
        r"(?i)(cổ\s*phần|joint[\s_-]*stock)": ("JointStockCompany", "Enterprise"),
        r"(?i)(tnhh|limited[\s_-]*liability|trách\s*nhiệm\s*hữu\s*hạn)": ("LimitedLiabilityCompany", "Enterprise"),
        r"(?i)(tư\s*nhân|private)": ("PrivateEnterprise", "Enterprise"),
        r"(?i)(hợp\s*danh|partnership)": ("Partnership", "Enterprise"),
        r"(?i)(hộ\s*kinh\s*doanh|household)": ("HouseholdBusiness", "Enterprise"),
        r"(?i)(bộ|ministry)": ("Ministry", "Government"),
        r"(?i)(cơ\s*quan|agency)": ("Agency", "Government"),
        r"(?i)(tòa\s*án|court)": ("Court", "Government"),
        r"(?i)(chính\s*phủ|government|nhà\s*nước)": ("Government", "Organization"),
        r"(?i)(tổ\s*chức|organization)": ("Organization", "Thing"),

        # Legal document patterns
        r"(?i)(luật|law)": ("Law", "LegalDocument"),
        r"(?i)(nghị\s*định|decree)": ("Decree", "LegalDocument"),
        r"(?i)(thông\s*tư|circular)": ("Circular", "LegalDocument"),
        r"(?i)(quyết\s*định|decision)": ("Decision", "LegalDocument"),
        r"(?i)(nghị\s*quyết|resolution)": ("Resolution", "LegalDocument"),
        r"(?i)(văn\s*bản|document|legal[\s_-]*doc)": ("LegalDocument", "Thing"),

        # Legal concept patterns
        r"(?i)(quyền|right)": ("Right", "LegalConcept"),
        r"(?i)(nghĩa\s*vụ|obligation|duty)": ("Obligation", "LegalConcept"),
        r"(?i)(hình\s*phạt|penalty|punishment)": ("Penalty", "LegalConcept"),
        r"(?i)(điều\s*kiện|condition|requirement)": ("Condition", "LegalConcept"),
        r"(?i)(hành\s*động|action)": ("Action", "LegalConcept"),
        r"(?i)(thủ\s*tục|procedure|process)": ("Procedure", "LegalConcept"),
        r"(?i)(khái\s*niệm|concept)": ("LegalConcept", "Thing"),

        # Person role patterns
        r"(?i)(đại\s*diện|representative)": ("LegalRepresentative", "PersonRole"),
        r"(?i)(cổ\s*đông|shareholder)": ("Shareholder", "PersonRole"),
        r"(?i)(thành\s*viên|member)": ("Member", "PersonRole"),
        r"(?i)(giám\s*đốc|director)": ("Director", "PersonRole"),
        r"(?i)(hội\s*đồng\s*quản\s*trị|board)": ("BoardMember", "PersonRole"),
        r"(?i)(người\s*sáng\s*lập|founder)": ("Founder", "PersonRole"),
        r"(?i)(vai\s*trò|role|person)": ("PersonRole", "Thing"),

        # Quantity patterns
        r"(?i)(tiền|monetary|money|vốn|capital)": ("Monetary", "Quantity"),
        r"(?i)(phần\s*trăm|percent|tỷ\s*lệ)": ("Percentage", "Quantity"),
        r"(?i)(thời\s*hạn|duration|period|term)": ("Duration", "Quantity"),
        r"(?i)(số\s*lượng|quantity|amount)": ("Quantity", "Thing"),
    }

    # Direct type name mappings (exact match)
    DIRECT_MAPPINGS = {
        "ORGANIZATION": ("Organization", "Thing"),
        "ENTERPRISE": ("Enterprise", "Organization"),
        "COMPANY": ("Enterprise", "Organization"),
        "JOINT_STOCK_COMPANY": ("JointStockCompany", "Enterprise"),
        "LIMITED_LIABILITY_COMPANY": ("LimitedLiabilityCompany", "Enterprise"),
        "LLC": ("LimitedLiabilityCompany", "Enterprise"),
        "GOVERNMENT": ("Government", "Organization"),
        "MINISTRY": ("Ministry", "Government"),
        "AGENCY": ("Agency", "Government"),
        "COURT": ("Court", "Government"),
        "LEGAL_DOCUMENT": ("LegalDocument", "Thing"),
        "LAW": ("Law", "LegalDocument"),
        "DECREE": ("Decree", "LegalDocument"),
        "CIRCULAR": ("Circular", "LegalDocument"),
        "DECISION": ("Decision", "LegalDocument"),
        "RESOLUTION": ("Resolution", "LegalDocument"),
        "LEGAL_TERM": ("LegalConcept", "Thing"),
        "LEGAL_CONCEPT": ("LegalConcept", "Thing"),
        "RIGHT": ("Right", "LegalConcept"),
        "OBLIGATION": ("Obligation", "LegalConcept"),
        "PENALTY": ("Penalty", "LegalConcept"),
        "CONDITION": ("Condition", "LegalConcept"),
        "ACTION": ("Action", "LegalConcept"),
        "PROCEDURE": ("Procedure", "LegalConcept"),
        "PERSON_ROLE": ("PersonRole", "Thing"),
        "ROLE": ("PersonRole", "Thing"),
        "SHAREHOLDER": ("Shareholder", "PersonRole"),
        "MEMBER": ("Member", "PersonRole"),
        "DIRECTOR": ("Director", "PersonRole"),
        "LEGAL_REFERENCE": ("LegalDocument", "Thing"),
        "QUANTITY": ("Quantity", "Thing"),
        "MONETARY": ("Monetary", "Quantity"),
        "PERCENTAGE": ("Percentage", "Quantity"),
        "DURATION": ("Duration", "Quantity"),
        "DATE": ("Duration", "Quantity"),
        "TIME": ("Duration", "Quantity"),
    }

    # Vietnamese labels for generated classes
    VIETNAMESE_LABELS = {
        "Thing": "Sự vật",
        "Organization": "Tổ chức",
        "Enterprise": "Doanh nghiệp",
        "JointStockCompany": "Công ty cổ phần",
        "LimitedLiabilityCompany": "Công ty TNHH",
        "SingleMemberLLC": "Công ty TNHH một thành viên",
        "MultiMemberLLC": "Công ty TNHH hai thành viên trở lên",
        "PrivateEnterprise": "Doanh nghiệp tư nhân",
        "Partnership": "Công ty hợp danh",
        "HouseholdBusiness": "Hộ kinh doanh",
        "Government": "Cơ quan nhà nước",
        "Ministry": "Bộ",
        "Agency": "Cơ quan",
        "Court": "Tòa án",
        "LegalDocument": "Văn bản pháp luật",
        "Law": "Luật",
        "Decree": "Nghị định",
        "Circular": "Thông tư",
        "Decision": "Quyết định",
        "Resolution": "Nghị quyết",
        "LegalConcept": "Khái niệm pháp lý",
        "Right": "Quyền",
        "Obligation": "Nghĩa vụ",
        "Penalty": "Hình phạt",
        "Condition": "Điều kiện",
        "Action": "Hành động",
        "Procedure": "Thủ tục",
        "PersonRole": "Vai trò cá nhân",
        "LegalRepresentative": "Người đại diện theo pháp luật",
        "Shareholder": "Cổ đông",
        "Member": "Thành viên",
        "Director": "Giám đốc",
        "BoardMember": "Thành viên HĐQT",
        "Founder": "Người sáng lập",
        "Quantity": "Số lượng",
        "Monetary": "Tiền tệ",
        "Percentage": "Tỷ lệ phần trăm",
        "Duration": "Thời hạn",
    }

    def __init__(self, base_ontology_path: Optional[str] = None):
        """
        Initialize the hierarchy builder.

        Args:
            base_ontology_path: Path to base ontology TTL file (optional)
        """
        self.base_classes: Set[str] = set()
        self.base_hierarchy: Dict[str, Optional[str]] = {}

        if base_ontology_path and Path(base_ontology_path).exists():
            self._load_base_classes(base_ontology_path)

    def _load_base_classes(self, path: str) -> None:
        """Load class names and hierarchy from base ontology."""
        try:
            from rdflib import Graph, RDFS, OWL

            g = Graph()
            g.parse(path, format="turtle")

            # Extract classes and their parents
            for s, p, o in g.triples((None, None, OWL.Class)):
                class_name = str(s).split("#")[-1]
                self.base_classes.add(class_name)

            for s, p, o in g.triples((None, RDFS.subClassOf, None)):
                child = str(s).split("#")[-1]
                parent = str(o).split("#")[-1]
                if child in self.base_classes:
                    self.base_hierarchy[child] = parent

            logger.info(f"Loaded {len(self.base_classes)} base classes from {path}")
        except Exception as e:
            logger.warning(f"Failed to load base ontology: {e}")

    def build_hierarchy(self, entities: List[Dict[str, Any]]) -> Dict[str, HierarchyMapping]:
        """
        Build class hierarchy from KG entities.

        Args:
            entities: List of entity dicts with 'type' or 'entity_type' field

        Returns:
            Dict mapping entity_type to HierarchyMapping
        """
        # Collect unique entity types
        entity_types: Dict[str, int] = {}
        for ent in entities:
            ent_type = ent.get("entity_type") or ent.get("type") or "Entity"
            entity_types[ent_type] = entity_types.get(ent_type, 0) + 1

        # Map each type to ontology class
        mappings: Dict[str, HierarchyMapping] = {}

        for ent_type, count in entity_types.items():
            mapping = self._infer_mapping(ent_type)
            mappings[ent_type] = mapping
            logger.debug(f"Mapped {ent_type} -> {mapping.class_name} (parent: {mapping.parent_class})")

        return mappings

    def _infer_mapping(self, entity_type: str) -> HierarchyMapping:
        """Infer class mapping for a single entity type."""
        # Normalize type name
        normalized = entity_type.upper().replace(" ", "_").replace("-", "_")

        # Try direct mapping first
        if normalized in self.DIRECT_MAPPINGS:
            class_name, parent = self.DIRECT_MAPPINGS[normalized]
            return HierarchyMapping(
                entity_type=entity_type,
                class_name=class_name,
                parent_class=parent,
                vietnamese_label=self.VIETNAMESE_LABELS.get(class_name, class_name),
                confidence=1.0,
            )

        # Try pattern matching
        for pattern, (class_name, parent) in self.TYPE_PATTERNS.items():
            if re.search(pattern, entity_type):
                return HierarchyMapping(
                    entity_type=entity_type,
                    class_name=class_name,
                    parent_class=parent,
                    vietnamese_label=self.VIETNAMESE_LABELS.get(class_name, class_name),
                    confidence=0.8,
                )

        # Check if type matches a base class
        for base_class in self.base_classes:
            if base_class.lower() in entity_type.lower():
                parent = self.base_hierarchy.get(base_class, "Thing")
                return HierarchyMapping(
                    entity_type=entity_type,
                    class_name=base_class,
                    parent_class=parent,
                    vietnamese_label=self.VIETNAMESE_LABELS.get(base_class, base_class),
                    confidence=0.7,
                )

        # Default: create new class under Thing
        class_name = self._normalize_class_name(entity_type)
        return HierarchyMapping(
            entity_type=entity_type,
            class_name=class_name,
            parent_class="Thing",
            vietnamese_label=entity_type,  # Use original as label
            confidence=0.5,
        )

    def _normalize_class_name(self, name: str) -> str:
        """Normalize entity type to PascalCase class name."""
        # Remove special characters and split
        words = re.split(r"[_\s-]+", name)
        # Capitalize each word
        return "".join(word.capitalize() for word in words if word)

    def get_hierarchy_dict(self, mappings: Dict[str, HierarchyMapping]) -> Dict[str, Optional[str]]:
        """Convert mappings to class -> parent dict for OntologyExpander."""
        hierarchy: Dict[str, Optional[str]] = {"Thing": None}

        for mapping in mappings.values():
            hierarchy[mapping.class_name] = mapping.parent_class

        return hierarchy

    def get_vietnamese_labels(self, mappings: Dict[str, HierarchyMapping]) -> Dict[str, str]:
        """Convert mappings to class -> Vietnamese label dict."""
        labels: Dict[str, str] = {"Thing": "Sự vật"}

        for mapping in mappings.values():
            labels[mapping.class_name] = mapping.vietnamese_label

        return labels
