"""
Pellet Reasoner Integration for Ontology Validation.

Uses owlready2's Pellet reasoner for symbolic OWL2 validation:
- Consistency checking (no logical contradictions)
- Satisfiability checking (classes can have instances)
- Inference of implicit facts

Designed for Vietnamese legal domain ontologies.
"""

import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Graceful import for owlready2
try:
    from owlready2 import (
        get_ontology,
        sync_reasoner_pellet,
        OwlReadyInconsistentOntologyError,
        JAVA_MEMORY,
    )
    OWLREADY2_AVAILABLE = True
except ImportError:
    OWLREADY2_AVAILABLE = False
    logger.warning("owlready2 not available, Pellet reasoning disabled")


@dataclass
class ReasoningResult:
    """Result from Pellet reasoning."""
    consistent: bool = True
    satisfiable: bool = True
    inferred_classes: List[str] = field(default_factory=list)
    inferred_properties: List[str] = field(default_factory=list)
    unsatisfiable_classes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    reasoning_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "consistent": self.consistent,
            "satisfiable": self.satisfiable,
            "inferred_classes": self.inferred_classes,
            "inferred_properties": self.inferred_properties,
            "unsatisfiable_classes": self.unsatisfiable_classes,
            "errors": self.errors,
            "warnings": self.warnings,
            "reasoning_time_ms": self.reasoning_time_ms,
        }


class PelletReasonerIntegration:
    """
    Pellet reasoner integration via owlready2.

    Provides symbolic OWL2 reasoning capabilities:
    - Consistency checking
    - Satisfiability checking
    - Property inference

    Example:
        >>> reasoner = PelletReasonerIntegration()
        >>> result = reasoner.reason("data/ontologies/base/legal-core.owl")
        >>> print(f"Consistent: {result.consistent}")
    """

    def __init__(
        self,
        timeout_seconds: int = 30,
        infer_property_values: bool = True,
        infer_data_property_values: bool = True,
        java_memory: int = 2000,
    ):
        """
        Initialize the Pellet reasoner integration.

        Args:
            timeout_seconds: Reasoning timeout in seconds
            infer_property_values: Whether to infer object property values
            infer_data_property_values: Whether to infer data property values
            java_memory: Java heap size in MB for Pellet
        """
        self.timeout = timeout_seconds
        self.infer_props = infer_property_values
        self.infer_data_props = infer_data_property_values
        self.java_memory = java_memory

        if OWLREADY2_AVAILABLE:
            # Configure Java memory for Pellet
            try:
                import owlready2
                owlready2.JAVA_MEMORY = java_memory
            except Exception as e:
                logger.warning(f"Failed to set Java memory: {e}")

    def reason(self, ontology_path: str) -> ReasoningResult:
        """
        Run Pellet reasoning on an ontology file.

        Args:
            ontology_path: Path to ontology file (TTL, OWL, RDF/XML)

        Returns:
            ReasoningResult with consistency, satisfiability, and inferences
        """
        if not OWLREADY2_AVAILABLE:
            return ReasoningResult(
                consistent=True,  # Assume true if can't check
                errors=["owlready2 not available, Pellet reasoning disabled"],
            )

        result = ReasoningResult()
        start = time.time()

        try:
            # Normalize path
            path = Path(ontology_path)
            if not path.exists():
                result.errors.append(f"Ontology file not found: {ontology_path}")
                return result

            # Load ontology
            onto_path = f"file://{path.absolute()}"
            onto = get_ontology(onto_path).load()

            # Run Pellet reasoner
            with onto:
                sync_reasoner_pellet(
                    infer_property_values=self.infer_props,
                    infer_data_property_values=self.infer_data_props,
                )

            # Check for unsatisfiable classes
            for cls in onto.classes():
                equiv = getattr(cls, "equivalent_to", [])
                if equiv and any("Nothing" in str(e) for e in equiv):
                    result.unsatisfiable_classes.append(cls.name)
                    result.satisfiable = False
                    result.warnings.append(
                        f"Unsatisfiable class: {cls.name}"
                    )

            result.consistent = True
            logger.info(f"Pellet reasoning completed successfully")

        except OwlReadyInconsistentOntologyError as e:
            result.consistent = False
            result.errors.append(f"Inconsistent ontology: {str(e)}")
            logger.warning(f"Ontology inconsistency detected: {e}")

        except Exception as e:
            error_msg = f"Reasoning error: {str(e)}"
            result.errors.append(error_msg)
            logger.error(error_msg)

        result.reasoning_time_ms = (time.time() - start) * 1000
        return result

    def reason_from_dict(self, ontology_dict: Dict[str, Any]) -> ReasoningResult:
        """
        Run Pellet reasoning on an ontology dict.

        Converts dict to temporary OWL file and runs reasoning.

        Args:
            ontology_dict: Ontology in dictionary format

        Returns:
            ReasoningResult
        """
        if not OWLREADY2_AVAILABLE:
            return ReasoningResult(
                consistent=True,
                errors=["owlready2 not available"],
            )

        try:
            # Import LegalOntology for conversion
            from vn_legal_rag.offline import LegalOntology

            # Convert dict to LegalOntology
            onto = LegalOntology.from_dict(ontology_dict)

            # Save to temp file
            with tempfile.NamedTemporaryFile(
                suffix=".ttl", delete=False, mode="w", encoding="utf-8"
            ) as f:
                f.write(onto.to_turtle())
                temp_path = f.name

            # Convert TTL to OWL for better owlready2 compatibility
            from rdflib import Graph
            g = Graph()
            g.parse(temp_path, format="turtle")

            owl_path = temp_path.replace(".ttl", ".owl")
            g.serialize(owl_path, format="xml")

            # Run reasoning
            result = self.reason(owl_path)

            # Cleanup temp files
            Path(temp_path).unlink(missing_ok=True)
            Path(owl_path).unlink(missing_ok=True)

            return result

        except Exception as e:
            return ReasoningResult(
                errors=[f"Failed to reason from dict: {str(e)}"]
            )

    def reason_from_ontology(self, ontology: "LegalOntology") -> ReasoningResult:
        """
        Run Pellet reasoning on a LegalOntology object.

        Args:
            ontology: LegalOntology instance

        Returns:
            ReasoningResult
        """
        return self.reason_from_dict(ontology.to_dict())

    @staticmethod
    def is_available() -> bool:
        """Check if Pellet reasoning is available."""
        return OWLREADY2_AVAILABLE
