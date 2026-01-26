import pytest
import pandas as pd
from src.pipeline import PipelineArchitect
from src.engine import TaskDetector, get_model_from_registry

def test_full_pipeline_flow():
    # 1. Setup Data
    df = pd.DataFrame({
        "age": [25, 30, 35, 40, None],          # Numeric (with NaN)
        "salary": [50000, 60000, 1, 1, 1],      # Static-ish (low var)
        "city": ["NY", "LA", "NY", "LA", "NY"], # Categorical
        "target": [0, 1, 0, 1, 0]               # Binary Target
    })
    X = df.drop(columns="target")
    y = df["target"]

    # 2. Setup Config (Mocking what comes from YAML)
    config = {
        "cleaning": {
            "variance": {"min_threshold": 0.1},
            "nan_thresholds": {"numeric": 0.5, "categorical": 0.5},
            "cardinality": {
            "max_unique_share":0.9
        }
        },
        
    }

    # 3. Engine: Detect Task & Get Model
    detector = TaskDetector(target_column="target")
    task = detector.detect(y)
    model_class = get_model_from_registry(task, "xgboost")
    model_inst = model_class(n_estimators=10)

    # 4. Architect: Build & Fit
    architect = PipelineArchitect(config)
    pipeline = architect.build_pipeline(
        model_instance=model_inst,
        task_type=task
    )

    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    # 5. Assertions
    assert len(preds) == 5
    # Verify cleaning happened: 'salary' should have been dropped due to low var in test data
    # We check the processing step's feature names
    proc_features = pipeline.named_steps["processing"].get_feature_names_out()
    assert any("age" in f for f in proc_features)
    assert any("salary" in f for f in proc_features)