import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn import metrics
from src.utils.logger import logger
 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import  mean_squared_error, accuracy_score
from sklearn.base import clone


import time
from src.engine import get_model_from_registry

class ModelEvaluator:
    """Calculates and stores performance metrics based on task type and config."""
    
    def __init__(self, task_type: str, metrics_list: List[str]):
        self.task_type = task_type
        self.metrics_list = metrics_list
        self.results: Dict[str, float] = {}

        # Metric Registry
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
        """Iterates through requested metrics and calculates scores."""
        logger.info(f"ModelEvaluator: Calculating {self.task_type} metrics...")
        
        for metric_name in self.metrics_list:
            if metric_name in self._metric_map:
                score = self._metric_map[metric_name](y_true, y_pred)
                self.results[metric_name] = round(float(score), 4)
            else:
                logger.warning(f"Metric '{metric_name}' not supported.")
        
        return self.results

class CrossValidator:
    """Handles professional CV strategies and generates OOF predictions."""
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state

    def _get_stratified_folds(self, X: pd.DataFrame, y: pd.Series, task_type: str):
        """Creates stratified folds for both classification and regression."""
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        if task_type == "regression":
            # Bin the target into 10 deciles to allow stratification
            # We use 'mids' to handle small datasets where some bins might have 1 member
            y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')
            return skf.split(X, y_binned)
        
        return skf.split(X, y)

    def run_cv(self, pipeline: Any, X: pd.DataFrame, y: pd.Series, task_type: str) -> Dict[str, Any]:
        """
        Executes CV, calculating mean/std loss and OOF predictions.
        Loss is MSE for regression and 1-Accuracy for classification.
        """
        logger.info(f"CrossValidator: Starting {self.n_splits}-fold CV for {task_type}")
        
        oof_preds = np.zeros(len(y))
        scores = []
        
        folds = self._get_stratified_folds(X, y, task_type)
        
        for fold, (train_idx, val_idx) in enumerate(folds):
            # Clone the pipeline to ensure a fresh state for every fold
            fold_pipe = clone(pipeline)
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            fold_pipe.fit(X_train, y_train)
            preds = fold_pipe.predict(X_val)
            
            oof_preds[val_idx] = preds
            
            # Calculate Fold Loss
            if task_type == "regression":
                score = mean_squared_error(y_val, preds)
            else:
                score = 1 - accuracy_score(y_val, preds) # Loss = Error Rate
            
            scores.append(score)
            logger.debug(f"Fold {fold+1}: Loss = {score:.4f}")

        return {
            "mean_loss": np.mean(scores),
            "std_loss": np.std(scores),
            "oof_predictions": oof_preds
        }
    

class LeaderboardEngine:
    def __init__(self, config, task_type, evaluator):
        self.config = config
        self.task_type = task_type
        self.evaluator = evaluator # The ModelEvaluator we built
        self.cv_engine = CrossValidator(n_splits=3)
        self.leaderboard_data = []

    def run_competition(self, pipeline_architect, X, y):
        """Runs CV for all models in config and returns a summary DataFrame."""
        model_slugs = self.config['model_selection']['models'][self.task_type]
        
        for slug in model_slugs:
            logger.info(f"Leaderboard: Evaluating {slug}...")
            
            # 1. Get Model & Build Pipeline
            model_class = get_model_from_registry(self.task_type, slug)
            pipeline = pipeline_architect.build_pipeline(model_class())
            
            # 2. Benchmark Time and CV
            start_time = time.time()
            cv_results = self.cv_engine.run_cv(pipeline, X, y, self.task_type)
            elapsed_time = time.time() - start_time
            
            # 3. Final Metrics using OOF Predictions
            # This is more robust than a single validation set
            final_metrics = self.evaluator.evaluate(y, cv_results['oof_predictions'])
            
            # 4. Compile Row
            row = {
                "Model": slug.upper(),
                "Train Time (s)": round(elapsed_time, 2),
                "CV Loss (Mean)": round(cv_results['mean_loss'], 4),
                "CV Loss (Std)": round(cv_results['std_loss'], 4),
                **final_metrics # Unpacks all metrics from config (Accuracy, F1, etc.)
            }
            self.leaderboard_data.append(row)
            
        return pd.DataFrame(self.leaderboard_data).sort_values(
            by="CV Loss (Mean)", ascending=True
        )