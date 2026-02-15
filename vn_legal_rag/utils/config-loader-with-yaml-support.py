"""
Configuration loader with YAML support.

Provides functions to load and validate configuration from YAML files.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/default.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config YAML file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.debug(f"Loading config from: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validate required sections
    validate_config(config)

    logger.debug(f"Config loaded successfully")
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If config is invalid
    """
    required_sections = ["llm", "database", "kg", "retrieval"]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Validate LLM config
    llm_config = config.get("llm", {})
    if "provider" not in llm_config:
        raise ValueError("Missing llm.provider in config")
    if "model" not in llm_config:
        raise ValueError("Missing llm.model in config")

    # Validate database config
    db_config = config.get("database", {})
    if "path" not in db_config:
        raise ValueError("Missing database.path in config")

    # Validate KG config
    kg_config = config.get("kg", {})
    if "path" not in kg_config:
        raise ValueError("Missing kg.path in config")

    # Validate dual level weights sum to 1.0
    dual_level_config = config.get("dual_level", {})
    if dual_level_config:
        weights = [
            dual_level_config.get("keyphrase_weight", 0),
            dual_level_config.get("semantic_weight", 0),
            dual_level_config.get("ppr_weight", 0),
            dual_level_config.get("concept_weight", 0),
            dual_level_config.get("theme_weight", 0),
            dual_level_config.get("hierarchy_weight", 0),
        ]
        total_weight = sum(weights)
        if not (0.99 <= total_weight <= 1.01):  # Allow small float errors
            logger.warning(
                f"Dual level weights sum to {total_weight:.3f}, not 1.0. "
                f"Weights will be normalized."
            )


def get_config_value(
    config: Dict[str, Any],
    key_path: str,
    default: Optional[Any] = None,
) -> Any:
    """
    Get nested config value using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., "llm.provider")
        default: Default value if key not found

    Returns:
        Config value or default

    Example:
        >>> config = {"llm": {"provider": "openai", "model": "gpt-4"}}
        >>> get_config_value(config, "llm.provider")
        'openai'
        >>> get_config_value(config, "llm.temperature", 0.7)
        0.7
    """
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def merge_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge two configs, with override_config taking precedence.

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value

    return merged


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    logger.info(f"Config saved to: {output_path}")


__all__ = [
    "load_config",
    "validate_config",
    "get_config_value",
    "merge_configs",
    "save_config",
]
