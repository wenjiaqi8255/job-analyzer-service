import logging
from .semantic_anomaly_detector import SemanticAnomalyDetector
from ..config import ModelThresholds, PipelineConfig, CacheConfig

logger = logging.getLogger(__name__)

class AnomalyDetectionPipeline:
    def __init__(self, embedding_model, semantic_baselines, boilerplate_baseline, text_preprocessor, nlp_model, thresholds: ModelThresholds, pipeline_config: PipelineConfig, cache_config: CacheConfig):
        self.semantic_anomaly_detector = SemanticAnomalyDetector(
            embedding_model=embedding_model,
            semantic_baselines=semantic_baselines,
            boilerplate_baseline=boilerplate_baseline,
            text_preprocessor=text_preprocessor,
            nlp_model=nlp_model,
            thresholds=thresholds,
            pipeline_config=pipeline_config,
            cache_config=cache_config
        )
        
    def run_anomaly_detection(self, job_description: str, analysis_result: dict) -> dict:
        """
        Runs the anomaly detection pipeline on a single job.

        Args:
            job_description (str): The effective description of the job.
            analysis_result (dict): The full analysis result from the AnalysisPipeline.

        Returns:
            dict: A dictionary containing detected anomalies and baseline composition.
        """
        primary_role = analysis_result.get('role', 'general')
        primary_industry = analysis_result.get('industry', 'unknown')
        
        logger.info(f"Detecting anomalies for role='{primary_role}' and industry='{primary_industry}'")

        anomaly_results = self.semantic_anomaly_detector.detect_anomalies(
            target_job=job_description,
            analysis_result=analysis_result
        )
        
        return anomaly_results 