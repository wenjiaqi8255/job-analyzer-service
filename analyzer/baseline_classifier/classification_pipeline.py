import logging
from .role_classifier import RoleClassifier
from .industry_classifier import IndustryClassifier
from ..config import AppConfig
from ..utils.embedding_processor import BatchEmbeddingProcessor

logger = logging.getLogger(__name__)

class ClassificationPipeline:
    def __init__(self, embedding_model, semantic_baselines, text_preprocessor, config: AppConfig):
        self.embedding_model = embedding_model
        self.semantic_baselines = semantic_baselines
        self.text_preprocessor = text_preprocessor
        self.config = config
        
        embedding_processor = BatchEmbeddingProcessor(embedding_model)
        
        self.role_classifier = RoleClassifier(
            embedding_processor=embedding_processor,
            semantic_baselines=self.semantic_baselines,
            text_preprocessor=self.text_preprocessor,
            thresholds=self.config.model_thresholds
        )
        
        self.industry_classifier = IndustryClassifier(
            embedding_processor=embedding_processor,
            baselines=self.semantic_baselines.get('industry', {}),
            preprocessor=self.text_preprocessor,
            threshold=self.config.model_thresholds.industry_similarity_threshold
        )
        
    def run(self, job_data: dict) -> dict:
        """
        Runs the classification pipeline on a single job.
        
        Args:
            job_data (dict): A dictionary containing job details like 'job_title', 
                             'effective_description', and 'industry'.
                             
        Returns:
            dict: A dictionary with classification results, e.g., {'role': 'Software Engineer', 'industry': 'Technology'}.
        """
        job_title = job_data.get('job_title', '')
        job_description = job_data.get('effective_description', '')
        job_industry = job_data.get('industry', '')
        company_name = job_data.get('company_name', '')
        
        classified_role = self.role_classifier.classify_job_role(job_title, job_description)
        classified_industry = self.industry_classifier.classify(job_industry, company_name, job_description)
        
        logger.info(f"Classification complete. Role='{classified_role}', Industry='{classified_industry}'")
        
        return {
            'role': classified_role,
            'industry': classified_industry if classified_industry != "Unknown" else job_industry
        } 