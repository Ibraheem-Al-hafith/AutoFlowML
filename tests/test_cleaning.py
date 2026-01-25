import pytest 
import pandas as pd
import numpy as np
from cleaning import VarianceStripper, UniversalDropper

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "high_nan": [1, None, None, None],      # 75% NaN
        "static": [1, 1, 1, 1],                # 0 variance
        "low_var": [1, 1, 1, 1.0001],          # Very low variance
        "good_num": [1, 2, 3, 4],              # Keep
        "cat_nan": ["A", "A", None, None],     # 50% NaN
        "good_cat": ["A", "B", "A", "B"]       # Keep
    })

def test_variace_stripper(sample_df):
    stripper = VarianceStripper(min_threshold=0.001)
    transformed = stripper.fit_transform(sample_df)

    # Should drop 'static' and 'low_var'
    assert "static" not in transformed.columns
    assert "low_var" not in transformed.columns
    assert "good_num" in transformed.columns
    assert stripper.n_features_in_ == 6
    assert np.array_equal(stripper.get_feature_names_out(),transformed.columns)

def test_universal_dropper(sample_df):
    thresholds = {"numeric": 0.5, "categorical": 0.3}
    dropper = UniversalDropper(thresholds=thresholds)
    transformed = dropper.fit_transform(sample_df)
    
    # 'high_nan' (75% > 50%) and 'cat_nan' (50% > 30%) should be dropped
    assert "high_nan" not in transformed.columns
    assert "cat_nan" not in transformed.columns
    assert "good_num" in transformed.columns
    assert len(dropper.get_feature_names_out()) == 4
    assert np.array_equal(dropper.get_feature_names_out(),transformed.columns)
