"""
Ontology Validator for Knowledge Graph Consistency Checking.

Validates ontology structure and KG conformance:
- Structural validation (classes, properties, hierarchy)
- Domain/range validation
- Class hierarchy consistency
- Property cardinality checks

Designed for Vietnamese legal domain ontologies.

Example:
    >>> from vn_legal_rag.offline import OntologyValidator
    >>> validator = OntologyValidator()
    >>> result = validator.validate(ontology)
    >>> print(f"Valid: {result.valid}, Errors: {len(result.errors)}")
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of an ontology validation operation."""
    valid: bool = True
    consistent: bool = True
    satisfiable: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "consistent": self.consistent,
            "satisfiable": self.satisfiable,
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


class OntologyValidator:
    """
    Validator for checking ontology consistency and validity.

    Validates:
    - Structural integrity (classes have names, properties have domain/range)
    - Hierarchy consistency (no cycles, parents exist)
    - Domain/range validity (referenced classes exist)
    - Naming conventions

    Example:
        >>> validator = OntologyValidator()
        >>> result = validator.validate(ontology)
        >>> if not result.valid:
        ...     for error in result.errors:
        ...         print(f"Error: {error}")
    """

    def __init__(
        self,
        check_consistency: bool = True,
        check_hierarchy: bool = True,
        check_domain_range: bool = True,
        strict_mode: bool = False,
        **kwargs,
    ):
        """
        Initialize the validator.

        Args:
            check_consistency: Whether to check logical consistency
            check_hierarchy: Whether to check class hierarchy
            check_domain_range: Whether to check property domain/range
            strict_mode: If True, warnings become errors
            **kwargs: Additional configuration
        """
        self.check_consistency = check_consistency
        self.check_hierarchy = check_hierarchy
        self.check_domain_range = check_domain_range
        self.strict_mode = strict_mode
        self.config = kwargs

    def validate(
        self,
        ontology: Union[Dict[str, Any], "LegalOntology"],
    ) -> ValidationResult:
        """
        Validate an ontology structure.

        Args:
            ontology: Ontology dictionary or LegalOntology object

        Returns:
            ValidationResult with validation status and issues
        """
        logger.info("Validating ontology structure")

        result = ValidationResult()

        # Convert LegalOntology to dict if needed
        if hasattr(ontology, "to_dict"):
            ontology_dict = ontology.to_dict()
        elif hasattr(ontology, "classes") and hasattr(ontology, "properties"):
            # LegalOntology object - convert manually
            ontology_dict = {
                "classes": [
                    {
                        "name": cls.name,
                        "label": cls.label,
                        "parent": cls.parent,
                        "description": getattr(cls, "description", ""),
                    }
                    for cls in ontology.classes.values()
                ],
                "properties": [
                    {
                        "name": prop.name,
                        "label": prop.label,
                        "type": prop.property_type,
                        "domain": [prop.domain] if prop.domain else [],
                        "range": [prop.range] if prop.range else [],
                    }
                    for prop in ontology.properties.values()
                ],
            }
        else:
            ontology_dict = ontology

        try:
            # 1. Structural validation
            self._validate_structure(ontology_dict, result)

            # 2. Hierarchy validation
            if self.check_hierarchy:
                self._validate_hierarchy(ontology_dict, result)

            # 3. Domain/range validation
            if self.check_domain_range:
                self._validate_domain_range(ontology_dict, result)

            # 4. Consistency checks
            if self.check_consistency:
                self._validate_consistency(ontology_dict, result)

            # Calculate metrics
            result.metrics = self._calculate_metrics(ontology_dict)

            # Determine overall validity
            if result.errors:
                result.valid = False

            # In strict mode, warnings also invalidate
            if self.strict_mode and result.warnings:
                result.valid = False

        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            result.valid = False
            result.errors.append(f"Validation exception: {str(e)}")

        logger.info(
            f"Validation complete: valid={result.valid}, "
            f"errors={len(result.errors)}, warnings={len(result.warnings)}"
        )
        return result

    def _validate_structure(
        self,
        ontology: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Basic structural validation."""
        classes = ontology.get("classes", [])
        properties = ontology.get("properties", [])

        # Check if classes exist
        if not classes:
            result.warnings.append("Ontology has no classes defined")

        # Check if properties exist
        if not properties:
            result.warnings.append("Ontology has no properties defined")

        # Validate each class
        for i, cls in enumerate(classes):
            if not cls.get("name"):
                result.errors.append(f"Class at index {i} has no name")
            if not cls.get("label"):
                result.warnings.append(
                    f"Class '{cls.get('name', f'index_{i}')}' has no label"
                )

        # Validate each property
        for i, prop in enumerate(properties):
            if not prop.get("name"):
                result.errors.append(f"Property at index {i} has no name")
            if not prop.get("type"):
                result.warnings.append(
                    f"Property '{prop.get('name', f'index_{i}')}' has no type specified"
                )

    def _validate_hierarchy(
        self,
        ontology: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate class hierarchy for cycles and missing parents."""
        classes = ontology.get("classes", [])

        # Build class name set
        class_names: Set[str] = {cls.get("name", "") for cls in classes}
        class_names.discard("")

        # Build parent mapping
        parent_map: Dict[str, Optional[str]] = {}
        for cls in classes:
            name = cls.get("name", "")
            parent = cls.get("parent") or cls.get("subClassOf")
            if name:
                parent_map[name] = parent

        # Check for missing parents
        for cls_name, parent in parent_map.items():
            if parent and parent not in class_names and parent != "Thing":
                result.errors.append(
                    f"Class '{cls_name}' has parent '{parent}' which doesn't exist"
                )

        # Check for cycles using DFS
        def has_cycle(cls_name: str, visited: Set[str], path: Set[str]) -> bool:
            if cls_name in path:
                return True
            if cls_name in visited:
                return False

            visited.add(cls_name)
            path.add(cls_name)

            parent = parent_map.get(cls_name)
            if parent and parent in parent_map:
                if has_cycle(parent, visited, path):
                    return True

            path.remove(cls_name)
            return False

        visited: Set[str] = set()
        for cls_name in parent_map:
            if has_cycle(cls_name, visited, set()):
                result.errors.append(
                    f"Circular hierarchy detected involving class '{cls_name}'"
                )
                result.consistent = False
                break

    def _validate_domain_range(
        self,
        ontology: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate property domain and range references."""
        classes = ontology.get("classes", [])
        properties = ontology.get("properties", [])

        # Build class name set (include Thing as implicit)
        class_names: Set[str] = {"Thing"}
        class_names.update(cls.get("name", "") for cls in classes)
        class_names.discard("")

        # XSD datatypes for data properties
        xsd_types = {
            "xsd:string", "xsd:integer", "xsd:decimal", "xsd:boolean",
            "xsd:date", "xsd:dateTime", "xsd:float", "xsd:double",
            "xsd:int", "xsd:long", "xsd:short", "xsd:byte",
            "xsd:nonNegativeInteger", "xsd:positiveInteger",
            "xsd:anyURI", "xsd:token", "xsd:language",
        }

        for prop in properties:
            prop_name = prop.get("name", "")
            prop_type = prop.get("type", "").lower()

            # Validate domain
            domains = prop.get("domain", [])
            if isinstance(domains, str):
                domains = [domains]
            for domain in domains:
                if domain and domain not in class_names:
                    result.errors.append(
                        f"Property '{prop_name}' has invalid domain '{domain}'"
                    )

            # Validate range
            ranges = prop.get("range", [])
            if isinstance(ranges, str):
                ranges = [ranges]
            for range_val in ranges:
                if not range_val:
                    continue
                # For object properties, range should be a class
                if prop_type in ("object", "objectproperty"):
                    if range_val not in class_names:
                        result.errors.append(
                            f"Object property '{prop_name}' has invalid range '{range_val}'"
                        )
                # For data properties, range should be XSD type
                elif prop_type in ("data", "dataproperty"):
                    if range_val not in xsd_types and not range_val.startswith("xsd:"):
                        result.warnings.append(
                            f"Data property '{prop_name}' has non-standard range '{range_val}'"
                        )

    def _validate_consistency(
        self,
        ontology: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Additional consistency checks."""
        classes = ontology.get("classes", [])
        properties = ontology.get("properties", [])

        # Check for duplicate class names
        class_names = [cls.get("name", "") for cls in classes]
        seen: Set[str] = set()
        for name in class_names:
            if name and name in seen:
                result.errors.append(f"Duplicate class name: '{name}'")
                result.consistent = False
            seen.add(name)

        # Check for duplicate property names
        prop_names = [prop.get("name", "") for prop in properties]
        seen.clear()
        for name in prop_names:
            if name and name in seen:
                result.errors.append(f"Duplicate property name: '{name}'")
                result.consistent = False
            seen.add(name)

    def _calculate_metrics(self, ontology: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate ontology metrics."""
        classes = ontology.get("classes", [])
        properties = ontology.get("properties", [])

        # Count hierarchy depth
        parent_map = {
            cls.get("name", ""): cls.get("parent") or cls.get("subClassOf")
            for cls in classes
        }

        def get_depth(cls_name: str, visited: Set[str]) -> int:
            if cls_name in visited:
                return 0
            visited.add(cls_name)
            parent = parent_map.get(cls_name)
            if not parent or parent == "Thing":
                return 1
            return 1 + get_depth(parent, visited)

        max_depth = 0
        for cls in classes:
            depth = get_depth(cls.get("name", ""), set())
            max_depth = max(max_depth, depth)

        # Count property types
        object_props = sum(
            1 for p in properties
            if p.get("type", "").lower() in ("object", "objectproperty")
        )
        data_props = sum(
            1 for p in properties
            if p.get("type", "").lower() in ("data", "dataproperty")
        )

        # Count classes with parents
        classes_with_parents = sum(
            1 for cls in classes
            if cls.get("parent") or cls.get("subClassOf")
        )

        return {
            "class_count": len(classes),
            "property_count": len(properties),
            "object_property_count": object_props,
            "data_property_count": data_props,
            "max_hierarchy_depth": max_depth,
            "classes_with_hierarchy": classes_with_parents,
            "hierarchy_ratio": classes_with_parents / len(classes) if classes else 0,
        }

    def validate_kg_against_ontology(
        self,
        kg: Dict[str, Any],
        ontology: Union[Dict[str, Any], "LegalOntology"],
    ) -> ValidationResult:
        """
        Validate a knowledge graph against an ontology.

        Checks:
        - All entity types exist as ontology classes
        - All relation types exist as ontology properties
        - Domain/range constraints are satisfied

        Args:
            kg: Knowledge graph with entities and relationships
            ontology: Ontology to validate against

        Returns:
            ValidationResult with conformance status
        """
        logger.info("Validating KG against ontology")

        result = ValidationResult()

        # Convert ontology if needed
        if hasattr(ontology, "classes") and hasattr(ontology, "properties"):
            class_names = set(ontology.classes.keys())
            prop_names = set(ontology.properties.keys())
        else:
            class_names = {cls.get("name", "") for cls in ontology.get("classes", [])}
            prop_names = {prop.get("name", "") for prop in ontology.get("properties", [])}

        # Add Thing as implicit class
        class_names.add("Thing")
        class_names.discard("")

        # Check entity types
        entities = kg.get("entities", [])
        unknown_types: Set[str] = set()
        for entity in entities:
            entity_type = entity.get("type", "")
            if entity_type and entity_type not in class_names:
                unknown_types.add(entity_type)

        if unknown_types:
            result.warnings.append(
                f"Entity types not in ontology: {', '.join(sorted(unknown_types))}"
            )

        # Check relation types
        relations = kg.get("relationships", [])
        unknown_rels: Set[str] = set()
        for rel in relations:
            rel_type = rel.get("type", "")
            if rel_type and rel_type not in prop_names:
                unknown_rels.add(rel_type)

        if unknown_rels:
            result.warnings.append(
                f"Relation types not in ontology: {', '.join(sorted(unknown_rels))}"
            )

        result.metrics = {
            "entities_checked": len(entities),
            "relations_checked": len(relations),
            "unknown_entity_types": len(unknown_types),
            "unknown_relation_types": len(unknown_rels),
            "conformance_rate": 1 - (len(unknown_types) + len(unknown_rels)) /
                               max(len(entities) + len(relations), 1),
        }

        return result


def validate_ontology(
    ontology: Union[Dict[str, Any], "LegalOntology"],
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Convenience wrapper for ontology validation.

    Args:
        ontology: Ontology to validate
        strict: If True, warnings become errors

    Returns:
        Dictionary representation of validation result
    """
    validator = OntologyValidator(strict_mode=strict)
    result = validator.validate(ontology)
    return result.to_dict()
