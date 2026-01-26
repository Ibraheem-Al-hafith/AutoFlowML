import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from src.utils.logger import logger
from src.engine import get_model_from_registry
import inspect

class ModelEvaluator:
    """Calculates metrics based on task type."""
    def __init__(self, task_type: str, metrics_list: List[str]):
        self.task_type = task_type
        self.metrics_list = metrics_list
        self._metric_map = {
            "accuracy": metrics.accuracy_score,
            "precision": lambda y, p: metrics.precision_score(y, p, average='weighted'),
            "recall": lambda y, p: metrics.recall_score(y, p, average='weighted'),
            "f1": lambda y, p: metrics.f1_score(y, p, average='weighted'),
            "rmse": lambda y, p: np.sqrt(metrics.mean_squared_error(y, p)),
            "mae": metrics.mean_absolute_error,
            "r2": metrics.r2_score
        }

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        results = {}
        for m in self.metrics_list:
            if m in self._metric_map:
                results[m] = round(float(self._metric_map[m](y_true, y_pred)), 4)
        return results

class CrossValidator:
    """Handles Stratified K-Fold and OOF generation."""
    def __init__(self, n_splits: int, random_state: int):
        self.n_splits = n_splits
        self.random_state = random_state

    def run_cv(self, pipeline: Any, X: pd.DataFrame, y: pd.Series, task_type: str) -> Dict[str, Any]:
        logger.info(f"CV: Starting {self.n_splits}-fold validation.")
        oof_preds = np.zeros(len(y))
        scores = []
        
        # Stratification Logic
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        target_split = y if task_type == "classification" else pd.qcut(y, q=min(10, len(y)//self.n_splits), labels=False, duplicates='drop')
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, target_split)):
            fold_pipe = clone(pipeline)
            fold_pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = fold_pipe.predict(X.iloc[val_idx])
            
            oof_preds[val_idx] = preds
            score = metrics.mean_squared_error(y.iloc[val_idx], preds) if task_type == "regression" else (1 - metrics.accuracy_score(y.iloc[val_idx], preds))
            scores.append(score)
            logger.info(f"   - Fold {fold+1} Loss: {score:.4f}")

        return {"mean_loss": np.mean(scores), "std_loss": np.std(scores), "oof_predictions": oof_preds}

class LeaderboardEngine:
    """Orchestrates the competition between models."""
    def __init__(self, config: Dict[str, Any], task_type: str, evaluator: ModelEvaluator):
        self.config = config
        self.task_type = task_type
        self.evaluator = evaluator
        self.cv_engine = CrossValidator(
            n_splits=config['settings']['cv_folds'], 
            random_state=config['settings']['random_state']
        )

    def run_competition(self, architect: Any, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        model_slugs = self.config['model_selection']['models'][self.task_type]
        leaderboard = []
        
        for slug in model_slugs:
            logger.info(f"Leaderboard: Processing {slug.upper()}")
            model_cls = get_model_from_registry(self.task_type, slug)
            
            # Init model with random_state if supported
            m_params = inspect.signature(model_cls).parameters
            model_inst = model_cls(random_state=self.config['settings']['random_state']) if 'random_state' in m_params else model_cls()
            
            pipeline = architect.build_pipeline(model_inst)
            
            start = time.time()
            cv_res = self.cv_engine.run_cv(pipeline, X, y, self.task_type)
            duration = time.time() - start
            
            metrics_res = self.evaluator.evaluate(y, cv_res['oof_predictions'])
            
            leaderboard.append({
                "Model": slug.upper(),
                "Time (s)": round(duration, 2),
                "CV Loss (Avg)": round(cv_res['mean_loss'], 4),
                **metrics_res
            })
        return pd.DataFrame(leaderboard).sort_values("CV Loss (Avg)")