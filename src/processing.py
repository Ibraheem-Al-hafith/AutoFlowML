from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, clone


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

class TargetEncodedModelWrapper(BaseEstimator):
    """
    Transparent proxy wrapper that handles target encoding/decoding 
    while exposing the underlying model's attributes.
    """
    def __init__(self, model, task_type="classification"):
        self.model = model
        self.task_type = task_type
        self.label_encoder = LabelEncoder()
        
    def fit(self, X, y):
        # We clone the model to ensure a fresh state
        self.model_ = clone(self.model)
        
        if self.task_type == "classification":
            y_encoded = self.label_encoder.fit_transform(y)
            self.classes_ = self.label_encoder.classes_
            self.model_.fit(X, y_encoded)
        else:
            self.model_.fit(X, y)
        return self

    def predict(self, X):
        preds = self.model_.predict(X)
        if self.task_type == "classification":
            return self.label_encoder.inverse_transform(preds)
        return preds

    def predict_proba(self, X):
        # Explicitly expose predict_proba if the underlying model has it
        if hasattr(self.model_, "predict_proba"):
            return self.model_.predict_proba(X)
        raise AttributeError(f"The underlying {type(self.model_).__name__} does not support predict_proba.")

    def __getattr__(self, name):
        """
        Proxy calls to the underlying model if they don't exist on the wrapper.
        This allows access to .feature_importances_, .coef_, etc.
        """
        if 'model_' in self.__dict__:
            return getattr(self.model_, name)
        # If model hasn't been fitted yet, check the template model
        return getattr(self.model, name)

    def __dir__(self):
        """Helper for tab-completion and inspection to show proxy attributes."""
        base_dir = list(object.__dir__(self))
        model_dir = dir(self.model_ if hasattr(self, 'model_') else self.model)
        return list(set(base_dir + model_dir))