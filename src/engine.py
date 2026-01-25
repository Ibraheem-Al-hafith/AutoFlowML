import pandas as pd
import numpy as np

from typing import Dict, Any, Optional

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from src.utils.logger import logger

# Defining Nested Model Registry:
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "classification": {
        "random_forest": RandomForestClassifier,
        "xgboost": XGBClassifier,
        "logistic": LogisticRegression,
        "lightgbm": LGBMClassifier,
        "tree": DecisionTreeClassifier,
    },
    "regression": {
        "random_forest": RandomForestRegressor,
        "xgboost": XGBRegressor,
        "ridge": Ridge,
        "lightgbm": LGBMRegressor,
        "tree": DecisionTreeRegressor,
    }
}

class TaskDetector:
    """Analyze target data to suggest ML task type and stores metadata"""
    def __init__(self, target_column: str, regression_threshold: int=15) -> None:
        self.target_column = target_column
        self.regression_threshold = regression_threshold
        self.metadata: Dict[str, Any] = {}
        self._suggested_task: Optional[str] = None
        
    def detect(self, y: pd.Series) -> str:
        """Heuristic-based task detection"""
        logger.info(f"TaskDetector: Analyzing target '{self.target_column}' ")

        unique_vals = y.nunique()
        y_type = y.dtype

        self.metadata = {
            "nunique": unique_vals,
            "dtype": str(y_type),
            "has_nans": y.isnull().any()
        }

        # Logic: Ig string/object/bool -> classification
        if pd.api.types.is_object_dtype(y) or pd.api.types.is_bool_dtype(y) or (y.dtype == 'category'):
            self._suggested_task = "classification"
        # Logic: If numeric but very few unique values:
        elif unique_vals <= self.regression_threshold:
            self._suggested_task = "classification"
        else:
            self._suggested_task = "regression"
        
        logger.info(f"TaskDetector: Detected {self._suggested_task.upper()}"
                    f"({unique_vals} unique values, dtype: {y_type})")
        return self._suggested_task
    
    @property
    def suggested_task(self) -> Optional[str]:
        return self._suggested_task



def get_model_from_registry(task: str, model_name: str) -> Any:
    """Safely retrieves a model class from the nested registry.
    Args:
        task(str): task of the model (classification, regression).
        model_name(str): model to be retrieved.
    returns:
        an initialized model
    """
    if task not in MODEL_REGISTRY:
        logger.error(f"Task '{task}' not supported in registry, supported task are: {str(MODEL_REGISTRY.keys())}")
        raise ValueError(f"Invalid task: {task}")
    
    models = MODEL_REGISTRY[task]
    if model_name not in models:
        logger.error(f"Model '{model_name}' no found for task '{task}', supported model for this task are {str(models.keys())}")
        raise KeyError(f"Model {model_name} misssing in {task} registry.")
    
    return models[model_name]

    