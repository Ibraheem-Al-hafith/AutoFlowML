import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from src.utils.logger import logger

class VarianceStripper(BaseEstimator, TransformerMixin):
    """Remove columns with zero variance or variance below a threshold"""
    def __init__(self, min_threshold: float = 0.0):
        self.min_threshold:float = min_threshold
        self.columns_to_keep_: List[str] = []
        self.n_features_in_: int = 0
    
    def fit(self, X:pd.DataFrame, y=None):
        self.n_features_in_ = X.shape[1]
        variances = X.var(numeric_only=True)

        # Identify low varinace columns
        low_var_cols = variances[variances <= self.min_threshold].index.tolist()

        # Identify static columns (even if non-numeric):
        static_cols = [col for col in X.columns if X[col].nunique() <= 1]

        to_drop = list(set(low_var_cols + static_cols))
        self.columns_to_keep_ = [c for c in X.columns.tolist() if c not in to_drop]

        msg: str = f"VarinaceStripper: Dropped {len(to_drop)} columns: {to_drop}" \
                    if to_drop else "VarianceStripper: No columns dropped."
        logger.info(msg)
        return self

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        return X[self.columns_to_keep_]
    
    def get_feature_names_out(self, input_features = None) -> np.ndarray:
        return np.array(self.columns_to_keep_)

class UniversalDropper(BaseEstimator, TransformerMixin):
    """Drop columns based on missing value threshold for numeric/categorical types"""
    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds
        self.columns_to_keep_: List[str] = []
        self.n_features_in_: int = 0
    
    def fit(self, X:pd.DataFrame, y=None):
        self.n_features_in_ = X.shape[1]
        to_drop = []

        for col in X.columns:
            null_ratio = X[col].isnull().mean()
            # Determine threshold based on dtype
            dtype_key = 'numeric' if pd.api.types.is_any_real_numeric_dtype(X[col]) else 'categorical'
            limit = self.thresholds.get(dtype_key, 0.5)

            if null_ratio > limit:
                to_drop.append(col)
        
        self.columns_to_keep_ = [c for c in X.columns if c not in to_drop]
        
        if to_drop:
            logger.info(f"UniversalDropper: Dropped{len(to_drop)} columns due to NaNs: {to_drop}")
        
        return self
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        return X[self.columns_to_keep_]
    
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.columns_to_keep_)
