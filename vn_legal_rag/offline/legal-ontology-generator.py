"""
Legal Ontology Generator for Vietnamese legal documents.

Generates OWL ontology from knowledge graph:
- Dynamic class/property inference from KG data
- LLM-powered Vietnamese label generation
- Turtle (.ttl) and JSON export

Designed for Vietnamese legal domain.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


# Vietnamese legal domain prompt for LLM
LEGAL_ONTOLOGY_PROMPT_VI = """Bạn là chuyên gia xây dựng ontology cho văn bản pháp luật Việt Nam.

Từ danh sách entities và relations được cung cấp, hãy tạo ontology với:

1. **Classes**: Phân loại entities thành các lớp có ý nghĩa
   - Mỗi class cần có: name (English, PascalCase), label (tiếng Việt), comment (mô tả)
   - Xây dựng hierarchy hợp lý (parent class)
   - Ví dụ: Organization, LegalDocument, PersonRole, LegalConcept, Quantity

2. **Properties**: Tạo properties từ relations
   - Object properties: liên kết giữa entities
   - Data properties: thuộc tính dữ liệu (text, number, date)
   - Mỗi property cần: name (camelCase), label (tiếng Việt), domain, range

3. **Quy tắc**:
   - Tên class/property bằng tiếng Anh (chuẩn OWL)
   - Label và comment bằng tiếng Việt
   - Hierarchy không quá 4 cấp
   - Tránh tạo class quá cụ thể (overfitting)

OUTPUT JSON:
{
  "classes": [
    {"name": "ClassName", "label": "Tên tiếng Việt", "comment": "Mô tả", "parent": "ParentClass"}
  ],
  "properties": [
    {"name": "propName", "type": "object|data", "label": "Tên tiếng Việt", "domain": ["Class"], "range": ["Class|xsd:string"]}
  ]
}
"""


@dataclass
class OntologyClass:
    """OWL class definition."""
    name: str
    label: str
    parent: Optional[str] = None
    description: str = ""
    uri: Optional[str] = None
    properties: List[str] = field(default_factory=list)


@dataclass
class OntologyProperty:
    """OWL property definition."""
    name: str
    label: str
    property_type: str  # "object" or "data"
    domain: str = "Thing"
    range: str = "Thing"
    description: str = ""
    uri: Optional[str] = None


@dataclass
class LegalOntology:
    """Legal domain ontology container."""
    classes: Dict[str, OntologyClass] = field(default_factory=dict)
    properties: Dict[str, OntologyProperty] = field(default_factory=dict)
    base_uri: str = "https://semantica.dev/legal/ontology#"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_class(self, cls: OntologyClass) -> None:
        """Add an ontology class."""
        self.classes[cls.name] = cls

    def add_property(self, prop: OntologyProperty) -> None:
        """Add an ontology property."""
        self.properties[prop.name] = prop

    def get_class_hierarchy(self) -> Dict[str, Optional[str]]:
        """Return class -> parent mapping for OntologyExpander."""
        return {cls.name: cls.parent for cls in self.classes.values()}

    def get_class_labels_vi(self) -> Dict[str, str]:
        """Return class -> Vietnamese label mapping."""
        return {cls.name: cls.label for cls in self.classes.values()}

    def to_turtle(self) -> str:
        """Export ontology to Turtle format."""
        lines = [
            f"@prefix : <{self.base_uri}> .",
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            "",
            "# Ontology declaration",
            f"<{self.base_uri}> a owl:Ontology ;",
            '    rdfs:label "Vietnamese Legal Document Ontology"@vi ;',
            '    rdfs:comment "Ontology for Vietnamese legal documents"@vi .',
            "",
        ]

        # Classes
        lines.append("# Classes")
        for cls in self.classes.values():
            lines.append(f":{cls.name} a owl:Class ;")
            lines.append(f'    rdfs:label "{cls.label}"@vi ;')
            if cls.parent:
                lines.append(f"    rdfs:subClassOf :{cls.parent} ;")
            if cls.description:
                desc = cls.description.replace('"', '\\"')
                lines.append(f'    rdfs:comment "{desc}"@vi ;')
            lines[-1] = lines[-1].rstrip(" ;") + " ."
            lines.append("")

        # Object Properties
        lines.append("# Object Properties")
        for prop in self.properties.values():
            prop_type = prop.property_type.lower()
            if prop_type in ("object", "objectproperty"):
                lines.append(f":{prop.name} a owl:ObjectProperty ;")
                lines.append(f'    rdfs:label "{prop.label}"@vi ;')
                lines.append(f"    rdfs:domain :{prop.domain} ;")
                lines.append(f"    rdfs:range :{prop.range} ;")
                if prop.description:
                    desc = prop.description.replace('"', '\\"')
                    lines.append(f'    rdfs:comment "{desc}"@vi ;')
                lines[-1] = lines[-1].rstrip(" ;") + " ."
                lines.append("")

        # Data Properties
        lines.append("# Data Properties")
        for prop in self.properties.values():
            prop_type = prop.property_type.lower()
            if prop_type in ("data", "dataproperty"):
                lines.append(f":{prop.name} a owl:DatatypeProperty ;")
                lines.append(f'    rdfs:label "{prop.label}"@vi ;')
                lines.append(f"    rdfs:domain :{prop.domain} ;")
                range_val = prop.range
                if range_val.startswith("xsd:"):
                    lines.append(f"    rdfs:range {range_val} ;")
                else:
                    lines.append(f"    rdfs:range :{range_val} ;")
                if prop.description:
                    desc = prop.description.replace('"', '\\"')
                    lines.append(f'    rdfs:comment "{desc}"@vi ;')
                lines[-1] = lines[-1].rstrip(" ;") + " ."
                lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "base_uri": self.base_uri,
            "classes": [
                {
                    "name": cls.name,
                    "label": cls.label,
                    "parent": cls.parent,
                    "description": cls.description,
                    "uri": cls.uri or f"{self.base_uri}{cls.name}",
                }
                for cls in self.classes.values()
            ],
            "properties": [
                {
                    "name": prop.name,
                    "label": prop.label,
                    "type": prop.property_type,
                    "domain": [prop.domain] if prop.domain else [],
                    "range": [prop.range] if prop.range else [],
                    "description": prop.description,
                    "uri": prop.uri or f"{self.base_uri}{prop.name}",
                }
                for prop in self.properties.values()
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LegalOntology":
        """Load ontology from dictionary."""
        ontology = cls(base_uri=data.get("base_uri", "https://semantica.dev/legal/ontology#"))
        ontology.metadata = data.get("metadata", {})

        for cls_data in data.get("classes", []):
            ontology.add_class(OntologyClass(
                name=cls_data["name"],
                label=cls_data.get("label", cls_data["name"]),
                parent=cls_data.get("parent"),
                description=cls_data.get("description", ""),
                uri=cls_data.get("uri"),
            ))

        for prop_data in data.get("properties", []):
            domain = prop_data.get("domain", ["Thing"])
            range_val = prop_data.get("range", ["Thing"])
            ontology.add_property(OntologyProperty(
                name=prop_data["name"],
                label=prop_data.get("label", prop_data["name"]),
                property_type=prop_data.get("type", "object"),
                domain=domain[0] if isinstance(domain, list) else domain,
                range=range_val[0] if isinstance(range_val, list) else range_val,
                description=prop_data.get("description", ""),
                uri=prop_data.get("uri"),
            ))

        return ontology

    @classmethod
    def from_json_file(cls, path: str) -> "LegalOntology":
        """Load ontology from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_json_file(self, path: str) -> None:
        """Save ontology to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class LegalOntologyGenerator:
    """
    Generate OWL ontology from legal knowledge graph.

    Features:
    - Dynamic class inference from KG entities
    - Dynamic property inference from KG relations
    - LLM-powered Vietnamese labeling (optional)
    - Turtle/JSON output

    Example:
        >>> generator = LegalOntologyGenerator(llm_provider=my_provider)
        >>> ontology = generator.generate_from_kg(kg)
        >>> turtle = ontology.to_turtle()
    """

    def __init__(
        self,
        base_uri: str = "https://semantica.dev/legal/ontology#",
        llm_provider: Optional[Any] = None,
        llm_model: str = "gpt-4o-mini",
        use_llm: bool = True,
        min_occurrences: int = 1,
    ):
        """
        Initialize the Ontology Generator.

        Args:
            base_uri: Base URI for the ontology
            llm_provider: LLM provider instance (must have generate_structured method)
            llm_model: LLM model name
            use_llm: Whether to use LLM for label generation
            min_occurrences: Minimum entity occurrences to create class
        """
        self.base_uri = base_uri
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.use_llm = use_llm and llm_provider is not None
        self.min_occurrences = min_occurrences

    def generate_from_kg(
        self,
        kg: Dict[str, Any],
        name: str = "VietnameseLegalOntology",
    ) -> LegalOntology:
        """
        Generate ontology from knowledge graph.

        Args:
            kg: Knowledge graph with entities/relationships
            name: Ontology name

        Returns:
            LegalOntology with inferred classes and properties
        """
        if self.use_llm and self.llm_provider:
            return self._generate_with_llm(kg, name)
        else:
            return self._generate_rule_based(kg, name)

    def _generate_with_llm(
        self,
        kg: Dict[str, Any],
        name: str,
    ) -> LegalOntology:
        """Generate ontology using LLM for Vietnamese labels."""
        logger.info("Generating ontology with LLM")

        entities_summary = self._summarize_entities(kg.get("entities", []))
        relations_summary = self._summarize_relations(kg.get("relationships", []))

        text = f"""# Entities (loại - số lượng):
{entities_summary}

# Relations (loại - số lượng):
{relations_summary}

Hãy tạo ontology phù hợp cho dữ liệu trên."""

        prompt = LEGAL_ONTOLOGY_PROMPT_VI + "\n\n" + text

        try:
            result = self.llm_provider.generate_structured(
                prompt,
                model=self.llm_model,
                temperature=0.2,
            )
            return self._parse_llm_output(result, name)
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}, falling back to rule-based")
            return self._generate_rule_based(kg, name)

    def _generate_rule_based(
        self,
        kg: Dict[str, Any],
        name: str,
    ) -> LegalOntology:
        """Generate ontology using rule-based inference."""
        logger.info("Generating ontology with rule-based inference")

        ontology = LegalOntology(base_uri=self.base_uri)

        # Add Thing as root class
        ontology.add_class(OntologyClass(
            name="Thing",
            label="Sự vật",
            parent=None,
            description="Lớp gốc của ontology",
        ))

        # Infer classes from entity types
        entity_types: Dict[str, int] = {}
        for ent in kg.get("entities", []):
            ent_type = ent.get("entity_type") or ent.get("type") or "Entity"
            entity_types[ent_type] = entity_types.get(ent_type, 0) + 1

        for ent_type, count in entity_types.items():
            if count >= self.min_occurrences and ent_type not in ontology.classes:
                ontology.add_class(OntologyClass(
                    name=ent_type,
                    label=ent_type,  # No Vietnamese label in rule-based
                    parent="Thing",
                    description=f"Entity type: {ent_type}",
                ))

        # Infer properties from relation types
        relation_types: Dict[str, int] = {}
        for rel in kg.get("relationships", []):
            rel_type = rel.get("relation_type") or rel.get("type") or "relatedTo"
            relation_types[rel_type] = relation_types.get(rel_type, 0) + 1

        for rel_type, count in relation_types.items():
            if count >= self.min_occurrences:
                ontology.add_property(OntologyProperty(
                    name=rel_type,
                    label=rel_type,
                    property_type="object",
                    domain="Thing",
                    range="Thing",
                ))

        # Add common data properties
        common_data_props = [
            ("hasConfidence", "Độ tin cậy", "Thing", "xsd:decimal"),
            ("hasText", "Nội dung", "Thing", "xsd:string"),
            ("hasSourceId", "ID nguồn", "Thing", "xsd:string"),
        ]
        for prop_name, label, domain, range_val in common_data_props:
            ontology.add_property(OntologyProperty(
                name=prop_name,
                label=label,
                property_type="data",
                domain=domain,
                range=range_val,
            ))

        ontology.metadata = {
            "name": name,
            "generated_with": "rule_based",
            "num_classes": len(ontology.classes),
            "num_properties": len(ontology.properties),
        }

        return ontology

    def _summarize_entities(self, entities: List[Dict]) -> str:
        """Summarize entities by type with counts."""
        type_counts: Dict[str, int] = {}
        type_examples: Dict[str, List[str]] = {}

        for ent in entities:
            ent_type = ent.get("type") or ent.get("entity_type") or "Unknown"
            ent_name = ent.get("name") or ent.get("text") or ""

            type_counts[ent_type] = type_counts.get(ent_type, 0) + 1
            if ent_type not in type_examples:
                type_examples[ent_type] = []
            if len(type_examples[ent_type]) < 3:
                type_examples[ent_type].append(ent_name)

        lines = []
        for ent_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            examples = ", ".join(type_examples.get(ent_type, [])[:3])
            lines.append(f"- {ent_type}: {count} (ví dụ: {examples})")

        return "\n".join(lines) if lines else "Không có entities"

    def _summarize_relations(self, relationships: List[Dict]) -> str:
        """Summarize relations by type with counts."""
        type_counts: Dict[str, int] = {}

        for rel in relationships:
            rel_type = rel.get("type") or rel.get("relationship_type") or "Unknown"
            type_counts[rel_type] = type_counts.get(rel_type, 0) + 1

        lines = []
        for rel_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"- {rel_type}: {count}")

        return "\n".join(lines) if lines else "Không có relations"

    def _parse_llm_output(
        self,
        result: Dict[str, Any],
        name: str,
    ) -> LegalOntology:
        """Parse LLM output to LegalOntology."""
        ontology = LegalOntology(base_uri=self.base_uri)

        # Add Thing as root class
        ontology.add_class(OntologyClass(
            name="Thing",
            label="Sự vật",
            parent=None,
            description="Lớp gốc của ontology",
        ))

        # Parse classes
        for cls_data in result.get("classes", []):
            cls_name = cls_data.get("name", "")
            if not cls_name or cls_name == "Thing":
                continue

            ontology.add_class(OntologyClass(
                name=cls_name,
                label=cls_data.get("label", cls_name),
                parent=cls_data.get("parent", "Thing"),
                description=cls_data.get("comment", ""),
                uri=f"{self.base_uri}{cls_name}",
            ))

        # Parse properties
        for prop_data in result.get("properties", []):
            prop_name = prop_data.get("name", "")
            if not prop_name:
                continue

            prop_type = prop_data.get("type", "object")
            domain = prop_data.get("domain", ["Thing"])
            range_val = prop_data.get("range", ["Thing"])

            domain_str = domain[0] if isinstance(domain, list) and domain else "Thing"
            range_str = range_val[0] if isinstance(range_val, list) and range_val else "Thing"

            ontology.add_property(OntologyProperty(
                name=prop_name,
                label=prop_data.get("label", prop_name),
                property_type=prop_type,
                domain=domain_str,
                range=range_str,
                description=prop_data.get("comment", ""),
                uri=f"{self.base_uri}{prop_name}",
            ))

        # Add common data properties
        common_data_props = [
            ("hasConfidence", "Độ tin cậy", "Thing", "xsd:decimal"),
            ("hasText", "Nội dung", "Thing", "xsd:string"),
            ("hasSourceId", "ID nguồn", "Thing", "xsd:string"),
        ]
        for prop_name, label, domain, range_val in common_data_props:
            if prop_name not in ontology.properties:
                ontology.add_property(OntologyProperty(
                    name=prop_name,
                    label=label,
                    property_type="data",
                    domain=domain,
                    range=range_val,
                ))

        ontology.metadata = {
            "name": name,
            "generated_with": "llm",
            "num_classes": len(ontology.classes),
            "num_properties": len(ontology.properties),
        }

        return ontology

    def validate_ontology(self, ontology: LegalOntology) -> Dict[str, Any]:
        """Validate ontology structure."""
        issues = []

        # Check class hierarchy
        for cls in ontology.classes.values():
            if cls.parent and cls.parent not in ontology.classes:
                issues.append({
                    "type": "missing_parent",
                    "class": cls.name,
                    "parent": cls.parent,
                })

        # Check property domains/ranges
        for prop in ontology.properties.values():
            prop_type = prop.property_type.lower()
            if prop_type in ("object", "objectproperty"):
                if prop.domain not in ontology.classes:
                    issues.append({
                        "type": "invalid_domain",
                        "property": prop.name,
                        "domain": prop.domain,
                    })
                if prop.range not in ontology.classes and not prop.range.startswith("xsd:"):
                    issues.append({
                        "type": "invalid_range",
                        "property": prop.name,
                        "range": prop.range,
                    })

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "num_classes": len(ontology.classes),
            "num_properties": len(ontology.properties),
        }
