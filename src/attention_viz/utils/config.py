"""Configuration utilities for attention visualization."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


class Config:
    """Configuration management for attention visualization."""
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            config_path = self._get_default_config_path()
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get path to default configuration file."""
        current_dir = Path(__file__).parent.parent.parent
        return str(current_dir / "configs" / "default.yaml")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "model": {
                "name": "bert-base-uncased",
                "device": "auto",
                "seed": 42,
            },
            "visualization": {
                "figsize": [12, 10],
                "cmap": "viridis",
                "dpi": 300,
                "save_format": "png",
            },
            "attention": {
                "max_length": 512,
                "discard_ratio": 0.9,
                "threshold": 0.1,
            },
            "evaluation": {
                "stability_method": "kendall_tau",
                "faithfulness_tests": ["deletion", "insertion", "sufficiency"],
                "interpretability_metrics": ["entropy", "sparsity", "symmetry", "concentration"],
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
            "paths": {
                "data_dir": "data",
                "assets_dir": "assets",
                "configs_dir": "configs",
                "logs_dir": "logs",
            },
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation).
            default: Default value if key not found.
            
        Returns:
            Configuration value.
        """
        try:
            return OmegaConf.select(self.config, key, default=default)
        except Exception:
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation).
            value: Value to set.
        """
        OmegaConf.set(self.config, key, value)
    
    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to file.
        
        Args:
            path: Path to save configuration. If None, uses current path.
        """
        save_path = path or self.config_path
        try:
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Configuration saved to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.get("paths.data_dir"),
            self.get("paths.assets_dir"),
            self.get("paths.configs_dir"),
            self.get("paths.logs_dir"),
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {directory}")


def get_config(config_path: Optional[str] = None) -> Config:
    """Get configuration instance.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration instance.
    """
    return Config(config_path)
