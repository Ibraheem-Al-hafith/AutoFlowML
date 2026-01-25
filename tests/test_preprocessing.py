import pytest
import pandas as pd
from src.processing import get_numeric_transformer, AutoDFColumnTransformer

def test_numeric_transformer_returns_df():
    df = pd.DataFrame({"age": [20, 30, None, 50]})
    pipe = get_numeric_transformer(scaler_type="standard")
    
    # We must call set_output on the pipeline or use it inside a ColumnTransformer
    pipe.set_output(transform="pandas")
    out = pipe.fit_transform(df)
    
    assert isinstance(out, pd.DataFrame)
    assert out["age"].isnull().sum() == 0  # Imputed
    assert out["age"].mean() < 1e-10       # Scaled (StandardScaler mean approx 0)

def test_auto_df_column_transformer():
    df = pd.DataFrame({
        "num": [1, 2, 3],
        "cat": ["a", "b", "a"]
    })
    
    from src.processing import get_categorical_transformer
    
    ct = AutoDFColumnTransformer(transformers=[
        ("n", get_numeric_transformer(), ["num"]),
        ("c", get_categorical_transformer(), ["cat"])
    ])
    
    out = ct.fit_transform(df)
    
    assert isinstance(out, pd.DataFrame)
    # Check if one-hot encoding created multiple columns
    assert any("cat" in col for col in out.columns)
    assert out.index.equals(df.index)