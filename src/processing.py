from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def get_numeric_transformer(scaler_type: str = "standard") -> Pipeline:
    """Returns a pipeline for numeric data: Imputation + Scaling."""
    scaler = RobustScaler() if scaler_type=='robust' else StandardScaler()
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler)
    ])

def get_categorical_transformer() -> Pipeline:
    """Returns a pipeline for categorical data: Imputation + encoding"""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

def get_text_transformer() -> TfidfVectorizer:
    """Return a transformer for text data."""
    return TfidfVectorizer(max_features=100)


class AutoDFColumnTransformer(ColumnTransformer):
    """
    Custom ColumnTransformer that enforces Pandas output by default.
    Explicitly defines parameters to remain compatible with sklearn introspection.
    """
    def __init__(
        self,
        transformers,
        *,
        remainder="drop",
        sparse_threshold=0.3,
        n_jobs=None,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=True,
    ):
        super().__init__(
            transformers=transformers,
            remainder=remainder,
            sparse_threshold=sparse_threshold,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights,
            verbose=verbose,
            verbose_feature_names_out=verbose_feature_names_out,
        )
        # The "magic" switch to ensure DataFrames are returned
        self.set_output(transform="pandas")