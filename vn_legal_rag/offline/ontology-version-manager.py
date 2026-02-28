"""
Ontology Version Manager for Multi-Domain Legal Ontologies.

Manages ontology versions and merging for Vietnamese legal domain:
- Load/merge multiple ontology sources (base, domain, generated)
- Timestamp-based generated ontology naming
- Priority-based merge (base < domain < generated)
- Export merged ontology for runtime use

Directory structure:
    data/ontologies/
    ├── base/          # Core classes (stable, git-tracked)
    ├── domains/       # Domain-specific ontologies
    ├── generated/     # Auto-generated with timestamp
    └── merged/        # Runtime merged ontology
"""

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class OntologyVersion:
    """Metadata for an ontology version."""
    path: Path
    source_type: str  # "base", "domain", "generated"
    timestamp: Optional[datetime] = None
    domain: Optional[str] = None
    priority: int = 0  # Higher = takes precedence in merge

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": str(self.path),
            "source_type": self.source_type,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "domain": self.domain,
            "priority": self.priority,
        }


class OntologyVersionManager:
    """
    Manage multi-domain ontology versions and merging.

    Features:
    - Load ontologies from base, domain, and generated directories
    - Merge with priority ordering (base < domain < generated)
    - Save generated ontologies with timestamp
    - Export merged ontology for runtime use

    Example:
        >>> vm = OntologyVersionManager()
        >>> versions = vm.list_versions()
        >>> merged = vm.merge(domains=["enterprise-law"])
        >>> merged.to_ttl_file("data/ontologies/merged/active.ttl")
    """

    DEFAULT_PRIORITIES = {
        "base": 0,
        "domain": 10,
        "generated": 20,
    }

    def __init__(self, ontologies_dir: str = "data/ontologies"):
        """
        Initialize the version manager.

        Args:
            ontologies_dir: Root directory for ontologies
        """
        self.root = Path(ontologies_dir)
        self.base_dir = self.root / "base"
        self.domains_dir = self.root / "domains"
        self.generated_dir = self.root / "generated"
        self.merged_dir = self.root / "merged"

        # Ensure directories exist
        for d in [self.base_dir, self.domains_dir, self.generated_dir, self.merged_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def list_versions(self) -> List[OntologyVersion]:
        """
        List all available ontology versions.

        Returns:
            List of OntologyVersion sorted by priority
        """
        versions = []

        # Base ontologies (lowest priority)
        for f in self.base_dir.glob("*.ttl"):
            versions.append(OntologyVersion(
                path=f,
                source_type="base",
                priority=self.DEFAULT_PRIORITIES["base"],
            ))
        for f in self.base_dir.glob("*.owl"):
            versions.append(OntologyVersion(
                path=f,
                source_type="base",
                priority=self.DEFAULT_PRIORITIES["base"],
            ))

        # Domain ontologies (medium priority)
        for f in self.domains_dir.glob("*.ttl"):
            domain = f.stem.replace("-", "_")
            versions.append(OntologyVersion(
                path=f,
                source_type="domain",
                domain=domain,
                priority=self.DEFAULT_PRIORITIES["domain"],
            ))

        # Generated ontologies (highest priority)
        for f in self.generated_dir.glob("kg-*.ttl"):
            if f.name == "kg-latest.ttl":
                continue  # Skip symlink/copy
            ts = self._parse_timestamp(f.stem)
            versions.append(OntologyVersion(
                path=f,
                source_type="generated",
                timestamp=ts,
                priority=self.DEFAULT_PRIORITIES["generated"],
            ))

        return sorted(versions, key=lambda v: v.priority)

    def save_generated(
        self,
        ontology: "LegalOntology",
        update_latest: bool = True,
    ) -> Path:
        """
        Save generated ontology with timestamp.

        Args:
            ontology: LegalOntology to save
            update_latest: If True, update kg-latest.ttl

        Returns:
            Path to saved ontology file
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        filename = f"kg-{timestamp}.ttl"
        path = self.generated_dir / filename

        ontology.to_ttl_file(str(path))
        logger.info(f"Saved generated ontology: {path}")

        if update_latest:
            latest = self.generated_dir / "kg-latest.ttl"
            shutil.copy(path, latest)
            logger.info(f"Updated kg-latest.ttl")

        return path

    def merge(
        self,
        domains: Optional[List[str]] = None,
        use_latest_generated: bool = True,
        output_path: Optional[str] = None,
    ) -> "LegalOntology":
        """
        Merge ontologies with priority: base < domain < generated.

        Args:
            domains: List of domain names to include (None = all)
            use_latest_generated: Include latest generated ontology
            output_path: Save merged ontology to path

        Returns:
            Merged LegalOntology
        """
        from vn_legal_rag.offline import LegalOntology

        merged = LegalOntology()
        versions = self.list_versions()

        # Filter by domains if specified
        if domains:
            versions = [
                v for v in versions
                if v.source_type != "domain" or v.domain in domains
            ]

        # Filter generated - use only latest if specified
        if use_latest_generated:
            latest = self.generated_dir / "kg-latest.ttl"
            if latest.exists():
                # Remove all generated versions except latest
                versions = [v for v in versions if v.source_type != "generated"]
                versions.append(OntologyVersion(
                    path=latest,
                    source_type="generated",
                    priority=self.DEFAULT_PRIORITIES["generated"],
                ))

        # Merge in priority order (lower priority first, so higher overwrites)
        for version in sorted(versions, key=lambda v: v.priority):
            if version.path.exists():
                try:
                    onto = self._load_ontology(version.path)
                    self._merge_into(merged, onto)
                    logger.info(f"Merged: {version.path}")
                except Exception as e:
                    logger.warning(f"Failed to merge {version.path}: {e}")

        # Save merged ontology
        if output_path:
            merged.to_ttl_file(output_path)
        else:
            # Default to active.ttl
            active_path = self.merged_dir / "active.ttl"
            merged.to_ttl_file(str(active_path))
            logger.info(f"Saved merged ontology to {active_path}")

        return merged

    def _load_ontology(self, path: Path) -> "LegalOntology":
        """Load ontology from file."""
        from vn_legal_rag.offline import LegalOntology

        suffix = path.suffix.lower()
        if suffix == ".ttl":
            return LegalOntology.from_ttl_file(str(path))
        elif suffix == ".json":
            return LegalOntology.from_json_file(str(path))
        elif suffix == ".owl":
            # Try to load OWL via TTL parser (works for most OWL files)
            return LegalOntology.from_ttl_file(str(path))
        else:
            raise ValueError(f"Unsupported ontology format: {suffix}")

    def _merge_into(self, target: "LegalOntology", source: "LegalOntology") -> None:
        """
        Merge source into target (source overwrites on conflict).

        Args:
            target: Target ontology to merge into
            source: Source ontology to merge from
        """
        for cls in source.classes.values():
            target.add_class(cls)
        for prop in source.properties.values():
            target.add_property(prop)

    def _parse_timestamp(self, stem: str) -> Optional[datetime]:
        """Parse timestamp from filename like kg-20260226-0100."""
        try:
            ts_str = stem.replace("kg-", "")
            return datetime.strptime(ts_str, "%Y%m%d-%H%M")
        except ValueError:
            return None

    def get_active(self) -> Optional["LegalOntology"]:
        """
        Get currently active merged ontology.

        Returns:
            LegalOntology from active.ttl or None if not exists
        """
        from vn_legal_rag.offline import LegalOntology

        active = self.merged_dir / "active.ttl"
        if active.exists():
            return LegalOntology.from_ttl_file(str(active))
        return None

    def cleanup_old_generated(self, keep_latest: int = 5) -> int:
        """
        Remove old generated ontologies, keeping latest N versions.

        Args:
            keep_latest: Number of latest versions to keep

        Returns:
            Number of files removed
        """
        generated = list(self.generated_dir.glob("kg-*.ttl"))
        generated = [f for f in generated if f.name != "kg-latest.ttl"]

        # Sort by timestamp (newest first)
        generated.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        removed = 0
        for f in generated[keep_latest:]:
            f.unlink()
            removed += 1
            logger.info(f"Removed old generated ontology: {f}")

        return removed
