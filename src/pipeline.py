from sklearn.pipeline import Pipeline
from src.cleaning import VarianceStripper, UniversalDropper
from src.processing import (
    get_numeric_transformer, 
    get_categorical_transformer, 
    AutoDFColumnTransformer
)
from src.utils.logger import logger
from typing import Dict, Any, List

class PipelineArchitect:
    """Assembles the cleaning, processing, and modeling blocks into one Pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def build_pipeline(
        self, 
        model_instance: Any, 
        numeric_features: List[str], 
        categorical_features: List[str]
    ) -> Pipeline:
        """Constructs the full end-to-end sklearn Pipeline."""
        
        logger.info("PipelineArchitect: Assembling components...")

        # 1. Cleaning Block
        cleaning_step = Pipeline(steps=[
            ("variance_stripper", VarianceStripper(
                min_threshold=self.config['cleaning']['variance']['min_threshold']
            )),
            ("nan_dropper", UniversalDropper(
                thresholds=self.config['cleaning']['nan_thresholds']
            ))
        ])

        # 2. Processing Block
        processing_step = AutoDFColumnTransformer(transformers=[
            ("num_pipe", get_numeric_transformer(), numeric_features),
            ("cat_pipe", get_categorical_transformer(), categorical_features)
        ])

        # 3. Final Assembly
        full_pipeline = Pipeline(steps=[
            ("cleaning", cleaning_step),
            ("processing", processing_step),
            ("model", model_instance)
        ])

        logger.info("PipelineArchitect: Pipeline built successfully.")
        return full_pipeline