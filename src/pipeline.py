from sklearn.pipeline import Pipeline
from src.cleaning import VarianceStripper, UniversalDropper
from src.processing import (
    get_numeric_transformer, 
    get_categorical_transformer, 
    AutoDFColumnTransformer
)

def build_full_pipeline(config, numeric_cols, categorical_cols, model):
    """
    Stitches the components together.
    """
    # 1. Cleaning Stage
    cleaner = Pipeline(steps=[
        ("variance", VarianceStripper(min_threshold=config['cleaning']['variance']['min_threshold'])),
        ("nan_dropper", UniversalDropper(thresholds=config['cleaning']['nan_thresholds']))
    ])

    # 2. Processing Stage (The Column Transformer)
    processor = AutoDFColumnTransformer(transformers=[
        ("num", get_numeric_transformer(), numeric_cols),
        ("cat", get_categorical_transformer(), categorical_cols)
    ])

    # 3. Final Pipeline
    full_pipe = Pipeline(steps=[
        ("cleaning", cleaner),
        ("processing", processor),
        ("model", model)
    ])
    
    return full_pipe