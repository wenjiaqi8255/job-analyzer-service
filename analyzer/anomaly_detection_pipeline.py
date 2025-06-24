import logging
from .semantic_anomaly_detector import SemanticAnomalyDetector
from .config import ModelThresholds, PipelineConfig, CacheConfig

logger = logging.getLogger(__name__)

class AnomalyDetectionPipeline:
    def __init__(self, embedding_model, semantic_baselines, text_preprocessor, nlp_model, thresholds: ModelThresholds, pipeline_config: PipelineConfig, cache_config: CacheConfig):
        self.semantic_anomaly_detector = SemanticAnomalyDetector(
            embedding_model=embedding_model,
            semantic_baselines=semantic_baselines,
            text_preprocessor=text_preprocessor,
            nlp_model=nlp_model,
            thresholds=thresholds,
            pipeline_config=pipeline_config,
            cache_config=cache_config
        )
        
    def run(self, job_data: dict, classification_result: dict) -> list:
        """
        Runs the anomaly detection pipeline on a single job.
        
        Args:
            job_data (dict): Dictionary with job details.
            classification_result (dict): Dictionary with 'role' and 'industry' from the classification pipeline.
            
        Returns:
            list: A list of detected anomalies.
        """
        description = job_data.get('effective_description', '')
        role = classification_result.get('role')
        industry = classification_result.get('industry')
        
        logger.info(f"Detecting anomalies for role='{role}' and industry='{industry}'")

        anomalies = self.semantic_anomaly_detector.detect_anomalies(
            target_job=description,
            role_baseline_name=role,
            industry_baseline_name=industry
        )
        
        return anomalies 