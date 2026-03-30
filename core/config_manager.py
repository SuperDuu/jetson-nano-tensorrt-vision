"""
Configuration Manager for RBC2026 Robocon Vision System.

This module provides configuration loading and validation functionality.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Manages application configuration from YAML file with relative path resolution."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Project root is the grandparent of this file (core/config_manager.py -> project_root)
        self.project_root = Path(__file__).resolve().parent.parent
        
        if config_path is None:
            # Default: config.yaml in the directory where the script is run
            config_path = "config.yaml"
        
        # If absolute, use as is. If relative, resolve against CWD first, then project root.
        test_path = Path(config_path)
        if test_path.is_absolute():
            self.config_path = test_path
        else:
            # Try current directory first, then root
            if test_path.exists():
                self.config_path = test_path.resolve()
            else:
                self.config_path = self.project_root / config_path
        
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            self._resolve_paths()
            self._validate_config()
            self.logger.info(f"Configuration loaded successfully from {self.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _resolve_paths(self) -> None:
        """
        Recursively resolve relative paths to absolute paths based on project root.
        Looks for strings that look like paths and resolves them if they exist relative to root.
        """
        def fix_paths(data):
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, (dict, list)):
                        fix_paths(v)
                    elif isinstance(v, str) and ('.' in v or '/' in v):
                        # Heuristic: if it's a string with path chars and not an absolute path
                        p = Path(v)
                        if not p.is_absolute():
                            # If it exists relative to project root, resolve it
                            potential_path = self.project_root / p
                            # We check common extensions to avoid resolving random strings
                            if p.suffix in ['.xml', '.bin', '.json', '.pt', '.yaml', '.txt', '.jpg', '.png']:
                                data[k] = str(potential_path)
            elif isinstance(data, list):
                for i in range(len(data)):
                    fix_paths(data[i])

        # Resolve paths in common keys
        if 'paths' in self.config: fix_paths(self.config['paths'])
        if 'models' in self.config: fix_paths(self.config['models'])
        if 'v1_model' in self.config: fix_paths(self.config['v1_model'])
        if 'v2_model' in self.config: fix_paths(self.config['v2_model'])
        if 'test_image_path' in self.config.get('system', {}):
             p = Path(self.config['system']['test_image_path'])
             if not p.is_absolute():
                 self.config['system']['test_image_path'] = str(self.project_root / p)

    def _validate_config(self) -> None:
        """Validate configuration values."""
        # Check basic structure
        if not self.config:
            raise ValueError("Configuration object is empty")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split('.')
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            return None # Return None if not found, consistent with original

    def get_path(self, key_path: str) -> str:
        """Returns the absolute path for a given config key."""
        return self.get(key_path)

    def __getitem__(self, key: str) -> Any:
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        return key in self.config
