import pytest
import yaml
from pathlib import Path
from ..src.utils.config_loader import load_config

def test_load_config_success(tmp_path):
    """Test that a valid YAML file is loaded correctly into a dictionary."""
    # --- ARRANGE ---
    # Create a temporary directory and a dummy config file
    d = tmp_path / "sub"
    d.mkdir()
    config_file = d / "test_config.yaml"
    
    dummy_data = {
        "metadata": {"project_name": "TestProject"},
        "cleaning": {"nan_thresholds": {"numeric": 0.5}}
    }
    
    with open(config_file, "w") as f:
        yaml.dump(dummy_data, f)

    # --- ACT ---
    loaded_cfg = load_config(config_file)

    # --- ASSERT ---
    assert loaded_cfg["metadata"]["project_name"] == "TestProject"
    assert loaded_cfg["cleaning"]["nan_thresholds"]["numeric"] == 0.5

def test_load_config_file_not_found():
    """Test that the loader raises FileNotFoundError for missing files."""
    invalid_path = Path("non_existent_path.yaml")
    
    with pytest.raises(FileNotFoundError):
        load_config(invalid_path)

def test_load_config_invalid_yaml(tmp_path):
    """Test that the loader raises an error for malformed YAML."""
    config_file = tmp_path / "bad_config.yaml"
    config_file.write_text("invalid: [yaml: : structure")

    with pytest.raises(Exception): # yaml.YAMLError
        load_config(config_file)