import pandas as pd
import numpy as np
from src.evaluation import CrossValidator

def test_cross_validator_regression():
    from sklearn.linear_model import LinearRegression
    X = pd.DataFrame({"feat": range(100)})
    y = pd.Series(range(100)) + np.random.normal(0, 0.1, 100)
    
    cv = CrossValidator(n_splits=3)
    results = cv.run_cv(LinearRegression(), X, y, task_type="regression")
    
    assert "mean_loss" in results
    assert len(results["oof_predictions"]) == 100
    assert results["std_loss"] >= 0