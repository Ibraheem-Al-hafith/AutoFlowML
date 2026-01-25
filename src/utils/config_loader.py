import yaml
from pathlib import Path
from typing import Any, Dict
from logger import logger

class ConfigNode(dict):
    """Dictionary -> object-style configuration tree with pretty-printing."""
    def __getattr__(self, key) -> Any:
        value = self.get(key)
        if isinstance(value, dict) and not isinstance(value, ConfigNode):
            # Wrap on the fly if it hasn't been wrapped yet
            value = ConfigNode(value)
        return value
    
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delattr__

    def __str__(self):
        return self._build_tree(self)

    def _build_tree(self, data, indent=""):
        lines = []
        items = list(data.items())
        for i, (key, value) in enumerate(items):
            is_last = (i == len(items) - 1)
            connector = "└── " if is_last else "├── "
            
            if isinstance(value, dict):
                lines.append(f"{indent}{connector}{key}")
                # Extend the vertical line if not at the end of a branch
                next_indent = indent + ("    " if is_last else "│   ")
                lines.append(self._build_tree(value, next_indent))
            else:
                lines.append(f"{indent}{connector}{key}: {value}")
                
        return "\n".join(lines)


def load_config(config_path: Path) -> ConfigNode:
    """
    Loads and validates the YAML configuration file.
    
    Args:
        config_path (Path): The filesystem path to the config.yaml file.
        
    Returns:
        ConfigNode : The parsed configuration data.
    """
    logger.info(f"Searching for configuration at: {config_path}")
    
    if not config_path.exists():
        logger.error(f"Configuration file not found at {config_path}")
        raise FileNotFoundError(f"Missing config file: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully.")
            return ConfigNode(config)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise e

if __name__ == "__main__":
    # Test block
    try:
        current_path = Path.cwd()
        # Adjusted path logic to find config.yaml in the root from src/utils
        root_path = current_path.parent.parent if "src" in str(current_path) else current_path
        cfg = load_config(root_path / "config.yaml")
        print(f"Loaded Project: {cfg.metadata.project_name}")
        print(cfg)
    except Exception as e:
        print(f"Failed to load config: {e}")