import pytest
import pandas as pd
from src.engine import TaskDetector, get_model_from_registry
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def test_task_detection_classification():
    detector = TaskDetector(target_column="target")
    
    # Test with string categories
    y_cat = pd.Series(["A", "B", "A", "A", "C"])
    assert detector.detect(y_cat) == "classification"
    
    # Test with low-cardinality integers
    y_int = pd.Series([0, 1, 0, 1, 0])
    assert detector.detect(y_int) == "classification"

    # Test with low-cardinality floats
    y_int = pd.Series([0.0, 1.0, 0, 1.0, 0.0])
    assert detector.detect(y_int) == "classification"

def test_task_detection_regression():
    detector = TaskDetector(target_column="target")
    
    # Test with high-cardinality floats
    y_cont = pd.Series(range(100)) + 0.5
    assert detector.detect(y_cont) == "regression"

def test_registry_retrieval():
    # Valid retrieval
    rf_class = get_model_from_registry("classification", "rf")
    assert rf_class == RandomForestClassifier
    
    # Invalid task
    with pytest.raises(ValueError):
        get_model_from_registry("clustering", "rf")
    
    # Invalid model name
    with pytest.raises(KeyError):
        get_model_from_registry("regression", "non_existent_model")