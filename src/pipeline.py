from sklearn.pipeline import Pipeline
from src.cleaning import VarianceStripper, UniversalDropper, CardinalityStripper
from src.processing import (
    get_numeric_transformer, 
    get_categorical_transformer, 
    AutoDFColumnTransformer,
    TargetEncodedModelWrapper
)
from sklearn.compose import make_column_selector
from src.utils.logger import logger
from typing import Dict, Any

class PipelineArchitect:
    """Assembles the cleaning, processing, and modeling blocks into one Pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def build_pipeline(
        self, 
        model_instance: Any, 
        task_type: str
    ) -> Pipeline:
        """Constructs the full end-to-end sklearn Pipeline."""
        
        logger.info("PipelineArchitect: Assembling components...")

        # 1. Cleaning Block
        # Inside PipelineArchitect.build_pipeline:

        cleaning_step = Pipeline(steps=[
            ("cardinality_stripper", CardinalityStripper(
                threshold=self.config['cleaning']['cardinality']['max_unique_share']
            )),
            ("variance_stripper", VarianceStripper(
                min_threshold=self.config['cleaning']['variance']['min_threshold']
            )),
            ("nan_dropper", UniversalDropper(
                thresholds=self.config['cleaning']['nan_thresholds']
            ))
        ])

        # 2. Processing Block
        processing_step = AutoDFColumnTransformer(transformers=[
            ("num_pipe", get_numeric_transformer(), make_column_selector(dtype_include=['number'])),
            ("cat_pipe", get_categorical_transformer(), make_column_selector(dtype_exclude=["number"]))
        ])

        # wrapping the model instance :
        wrapped_model = TargetEncodedModelWrapper(model_instance, task_type=task_type)

        # 3. Final Assembly
        full_pipeline = Pipeline(steps=[
            ("cleaning", cleaning_step),
            ("processing", processing_step),
            ("model", wrapped_model)
        ])

        logger.info("PipelineArchitect: Pipeline built successfully.")
        return full_pipeline
    
