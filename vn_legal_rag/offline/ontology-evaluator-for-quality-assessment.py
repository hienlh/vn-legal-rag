"""
Ontology Evaluator for Quality Assessment.

Evaluates ontologies for quality, coverage, and completeness:
- Competency question validation
- Coverage and completeness metrics
- Gap identification
- Class granularity evaluation
- Relation completeness checking
- Improvement suggestions

Designed for Vietnamese legal domain ontologies.

Example:
    >>> from vn_legal_rag.offline import OntologyEvaluator
    >>> evaluator = OntologyEvaluator()
    >>> result = evaluator.evaluate_ontology(ontology, competency_questions=[
    ...     "Ai là người đại diện theo pháp luật của công ty?",
    ...     "Công ty cổ phần có những loại hình nào?",
    ... ])
    >>> print(f"Coverage: {result.coverage_score:.2%}")
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CompetencyQuestion:
    """
    Competency question for ontology validation.

    Represents a question the ontology should be able to answer.

    Attributes:
        question: Question text (Vietnamese or English)
        category: Question category (general, organizational, temporal, legal)
        priority: Priority level (1=high, 2=medium, 3=low)
        answerable: Whether ontology can answer this question
        trace_to_elements: Ontology elements related to this question
    """
    question: str
    category: str = "general"
    priority: int = 1
    answerable: bool = False
    trace_to_elements: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Ontology evaluation result."""
    coverage_score: float
    completeness_score: float
    gaps: List[str]
    suggestions: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "coverage_score": round(self.coverage_score, 4),
            "completeness_score": round(self.completeness_score, 4),
            "gaps": self.gaps,
            "suggestions": self.suggestions,
            "metrics": self.metrics,
        }


class CompetencyQuestionsManager:
    """
    Manager for competency questions.

    Competency questions define what an ontology should be able to answer,
    serving as functional requirements for ontology design.

    Example:
        >>> manager = CompetencyQuestionsManager()
        >>> manager.add_question(
        ...     "Ai là người đại diện theo pháp luật?",
        ...     category="legal"
        ... )
        >>> results = manager.validate_ontology(ontology)
    """

    def __init__(self):
        """Initialize competency questions manager."""
        self.questions: List[CompetencyQuestion] = []

    def add_question(
        self,
        question: str,
        category: str = "general",
        priority: int = 1,
        **metadata,
    ) -> CompetencyQuestion:
        """
        Add a competency question.

        Args:
            question: Question text
            category: Category (general, organizational, temporal, legal)
            priority: Priority (1=high, 2=medium, 3=low)
            **metadata: Additional metadata

        Returns:
            Created CompetencyQuestion
        """
        cq = CompetencyQuestion(
            question=question,
            category=category,
            priority=priority,
            metadata=metadata,
        )
        self.questions.append(cq)
        logger.debug(f"Added competency question: {question[:50]}...")
        return cq

    def validate_ontology(self, ontology: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate ontology against competency questions.

        Args:
            ontology: Ontology dictionary

        Returns:
            Validation results with counts and breakdown
        """
        results = {
            "total_questions": len(self.questions),
            "answerable": 0,
            "unanswerable": 0,
            "by_category": {},
            "by_priority": {},
        }

        for question in self.questions:
            answerable = self._can_ontology_answer(ontology, question)
            question.answerable = answerable

            if answerable:
                results["answerable"] += 1
            else:
                results["unanswerable"] += 1

            # Track by category
            category = question.category
            if category not in results["by_category"]:
                results["by_category"][category] = {"answerable": 0, "unanswerable": 0}
            key = "answerable" if answerable else "unanswerable"
            results["by_category"][category][key] += 1

            # Track by priority
            priority = str(question.priority)
            if priority not in results["by_priority"]:
                results["by_priority"][priority] = {"answerable": 0, "unanswerable": 0}
            results["by_priority"][priority][key] += 1

        return results

    def _can_ontology_answer(
        self,
        ontology: Dict[str, Any],
        question: CompetencyQuestion,
    ) -> bool:
        """Check if ontology can answer the question using keyword matching."""
        # Clean question text
        question_text = question.question.lower()
        # Remove Vietnamese diacritics for comparison
        import unicodedata
        question_normalized = unicodedata.normalize("NFD", question_text)
        question_clean = "".join(
            c for c in question_normalized
            if unicodedata.category(c) != "Mn"
        )

        # Extract keywords (words > 3 chars)
        keywords = [w for w in question_clean.split() if len(w) > 3]

        # Check classes
        for cls in ontology.get("classes", []):
            cls_name = cls.get("name", "").lower()
            cls_label = cls.get("label", "").lower()
            for keyword in keywords:
                if keyword in cls_name or keyword in cls_label:
                    return True

        # Check properties
        for prop in ontology.get("properties", []):
            prop_name = prop.get("name", "").lower()
            prop_label = prop.get("label", "").lower()
            for keyword in keywords:
                if keyword in prop_name or keyword in prop_label:
                    return True

        return False

    def trace_to_elements(
        self,
        question: CompetencyQuestion,
        ontology: Dict[str, Any],
    ) -> List[str]:
        """Trace question to relevant ontology elements."""
        elements = []
        question_text = question.question.lower()

        # Extract keywords
        keywords = [w for w in question_text.split() if len(w) > 3]

        # Find matching classes
        for cls in ontology.get("classes", []):
            cls_name = cls.get("name", "")
            cls_label = cls.get("label", "").lower()
            for keyword in keywords:
                if keyword in cls_name.lower() or keyword in cls_label:
                    elements.append(f"class:{cls_name}")
                    break

        # Find matching properties
        for prop in ontology.get("properties", []):
            prop_name = prop.get("name", "")
            prop_label = prop.get("label", "").lower()
            for keyword in keywords:
                if keyword in prop_name.lower() or keyword in prop_label:
                    elements.append(f"property:{prop_name}")
                    break

        question.trace_to_elements = elements
        return elements

    def clear(self) -> None:
        """Clear all questions."""
        self.questions.clear()


class OntologyEvaluator:
    """
    Ontology evaluation engine for quality assessment.

    Features:
    - Validate ontology against competency questions
    - Calculate coverage and completeness scores
    - Identify gaps in ontology
    - Evaluate class granularity
    - Check relation completeness
    - Generate improvement suggestions

    Example:
        >>> evaluator = OntologyEvaluator()
        >>> result = evaluator.evaluate_ontology(ontology, competency_questions=[
        ...     "Công ty cổ phần cần bao nhiêu cổ đông?",
        ...     "Thủ tục thành lập doanh nghiệp là gì?",
        ... ])
        >>> print(result.to_dict())
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ontology evaluator.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.cq_manager = CompetencyQuestionsManager()

    def evaluate_ontology(
        self,
        ontology: Dict[str, Any],
        competency_questions: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """
        Evaluate ontology against competency questions.

        Args:
            ontology: Ontology dictionary
            competency_questions: Optional list of questions to validate

        Returns:
            EvaluationResult with scores, gaps, and suggestions
        """
        logger.info("Evaluating ontology quality")

        # Clear and add new questions
        self.cq_manager.clear()
        if competency_questions:
            for q in competency_questions:
                self.cq_manager.add_question(q)

        # Validate against competency questions
        validation = self.cq_manager.validate_ontology(ontology)

        # Calculate coverage score
        total = validation.get("total_questions", 0)
        answerable = validation.get("answerable", 0)
        coverage_score = answerable / total if total > 0 else 1.0

        # Calculate completeness score
        completeness_score = self._calculate_completeness(ontology)

        # Identify gaps
        gaps = self._identify_gaps(ontology, validation)

        # Generate suggestions
        suggestions = self._generate_suggestions(ontology, gaps)

        # Calculate metrics
        metrics = self._calculate_metrics(ontology, validation)

        result = EvaluationResult(
            coverage_score=coverage_score,
            completeness_score=completeness_score,
            gaps=gaps,
            suggestions=suggestions,
            metrics=metrics,
        )

        logger.info(
            f"Evaluation complete: coverage={coverage_score:.2%}, "
            f"completeness={completeness_score:.2%}"
        )
        return result

    def _calculate_completeness(self, ontology: Dict[str, Any]) -> float:
        """Calculate ontology completeness score."""
        classes = ontology.get("classes", [])
        properties = ontology.get("properties", [])

        if not classes and not properties:
            return 0.0

        # Check class completeness
        classes_complete = 0
        for cls in classes:
            has_name = bool(cls.get("name"))
            has_label = bool(cls.get("label"))
            # URI is optional, name+label is sufficient
            if has_name and has_label:
                classes_complete += 1

        classes_score = classes_complete / len(classes) if classes else 0.0

        # Check property completeness
        props_complete = 0
        for prop in properties:
            has_name = bool(prop.get("name"))
            has_type = bool(prop.get("type"))
            if has_name and has_type:
                props_complete += 1

        props_score = props_complete / len(properties) if properties else 0.0

        return (classes_score + props_score) / 2.0

    def _identify_gaps(
        self,
        ontology: Dict[str, Any],
        validation: Dict[str, Any],
    ) -> List[str]:
        """Identify gaps in ontology coverage."""
        gaps = []

        # Check unanswerable questions
        unanswerable = validation.get("unanswerable", 0)
        if unanswerable > 0:
            gaps.append(f"{unanswerable} competency questions cannot be answered")

        classes = ontology.get("classes", [])
        properties = ontology.get("properties", [])

        # Check for missing classes
        if not classes:
            gaps.append("Ontology has no classes")

        # Check for missing properties
        if not properties:
            gaps.append("Ontology has no properties")

        # Check for classes without properties
        classes_with_props = set()
        for prop in properties:
            domains = prop.get("domain", [])
            if isinstance(domains, str):
                domains = [domains]
            classes_with_props.update(domains)

        for cls in classes:
            cls_name = cls.get("name", "")
            if cls_name and cls_name not in classes_with_props and cls_name != "Thing":
                gaps.append(f"Class '{cls_name}' has no associated properties")

        # Check for classes without Vietnamese labels
        for cls in classes:
            cls_name = cls.get("name", "")
            cls_label = cls.get("label", "")
            if cls_name and not cls_label:
                gaps.append(f"Class '{cls_name}' has no Vietnamese label")

        return gaps

    def _generate_suggestions(
        self,
        ontology: Dict[str, Any],
        gaps: List[str],
    ) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []

        # Suggest based on gaps
        if any("competency questions" in gap for gap in gaps):
            suggestions.append(
                "Add classes/properties to answer unanswerable competency questions"
            )

        if any("no classes" in gap for gap in gaps):
            suggestions.append("Infer classes from entity types in your KG data")

        if any("no properties" in gap for gap in gaps):
            suggestions.append("Infer properties from relation types in your KG data")

        if any("no associated properties" in gap for gap in gaps):
            suggestions.append(
                "Define domain for properties to associate them with classes"
            )

        if any("no Vietnamese label" in gap for gap in gaps):
            suggestions.append(
                "Add Vietnamese labels for all classes using LLM generation"
            )

        classes = ontology.get("classes", [])

        # Check class count
        if len(classes) > 50:
            suggestions.append(
                "Consider splitting ontology into modules for better organization"
            )

        # Check hierarchy
        classes_with_parents = sum(
            1 for c in classes
            if c.get("subClassOf") or c.get("parent")
        )
        if classes and classes_with_parents < len(classes) * 0.3:
            suggestions.append(
                "Consider adding more hierarchical relationships between classes"
            )

        return suggestions

    def _calculate_metrics(
        self,
        ontology: Dict[str, Any],
        validation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        classes = ontology.get("classes", [])
        properties = ontology.get("properties", [])

        object_props = sum(
            1 for p in properties
            if p.get("type", "").lower() in ("object", "objectproperty")
        )
        data_props = sum(
            1 for p in properties
            if p.get("type", "").lower() in ("data", "dataproperty")
        )
        classes_with_hierarchy = sum(
            1 for c in classes
            if c.get("subClassOf") or c.get("parent")
        )

        total_questions = validation.get("total_questions", 0)
        answerable = validation.get("answerable", 0)

        return {
            "class_count": len(classes),
            "property_count": len(properties),
            "object_property_count": object_props,
            "data_property_count": data_props,
            "classes_with_hierarchy": classes_with_hierarchy,
            "hierarchy_ratio": classes_with_hierarchy / len(classes) if classes else 0,
            "competency_question_coverage": answerable / total_questions
                if total_questions > 0 else 1.0,
        }

    def evaluate_class_granularity(
        self,
        ontology: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate class granularity and suggest generalizations.

        Args:
            ontology: Ontology dictionary

        Returns:
            Granularity evaluation with suggestions
        """
        classes = ontology.get("classes", [])

        # Count instances per class if available
        instance_counts = {}
        for cls in classes:
            count = cls.get("entity_count", cls.get("inferred_count", 0))
            instance_counts[cls.get("name", "")] = count

        # Generate suggestions
        suggestions = []
        for cls_name, count in instance_counts.items():
            if count > 0 and count < 2:
                suggestions.append(
                    f"Class '{cls_name}' has very few instances - consider merging"
                )
            elif count > 1000:
                suggestions.append(
                    f"Class '{cls_name}' has many instances - consider splitting"
                )

        return {
            "instance_distribution": instance_counts,
            "suggestions": suggestions,
        }

    def evaluate_relation_completeness(
        self,
        ontology: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Check relation completeness.

        Args:
            ontology: Ontology dictionary

        Returns:
            Completeness evaluation with isolated classes
        """
        classes = ontology.get("classes", [])
        properties = ontology.get("properties", [])

        # Find classes with relations (as domain or range)
        classes_with_relations = set()
        for prop in properties:
            domains = prop.get("domain", [])
            ranges = prop.get("range", [])
            if isinstance(domains, str):
                domains = [domains]
            if isinstance(ranges, str):
                ranges = [ranges]
            classes_with_relations.update(domains)
            classes_with_relations.update(ranges)

        # Find isolated classes
        isolated_classes = [
            cls.get("name", "")
            for cls in classes
            if cls.get("name", "") not in classes_with_relations
            and cls.get("name", "") != "Thing"
        ]

        return {
            "classes_with_relations": len(classes_with_relations),
            "isolated_classes": isolated_classes,
            "relation_coverage": len(classes_with_relations) / len(classes)
                if classes else 0.0,
        }

    def generate_report(
        self,
        ontology: Dict[str, Any],
        competency_questions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.

        Args:
            ontology: Ontology dictionary
            competency_questions: Optional questions to validate

        Returns:
            Complete evaluation report
        """
        evaluation = self.evaluate_ontology(ontology, competency_questions)
        granularity = self.evaluate_class_granularity(ontology)
        completeness = self.evaluate_relation_completeness(ontology)

        return {
            "evaluation": evaluation.to_dict(),
            "granularity": granularity,
            "relation_completeness": completeness,
            "generated_at": datetime.now().isoformat(),
        }


def evaluate_ontology(
    ontology: Dict[str, Any],
    competency_questions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper for ontology evaluation.

    Args:
        ontology: Ontology to evaluate
        competency_questions: Optional questions to validate

    Returns:
        Evaluation result as dictionary
    """
    evaluator = OntologyEvaluator()
    result = evaluator.evaluate_ontology(ontology, competency_questions)
    return result.to_dict()
