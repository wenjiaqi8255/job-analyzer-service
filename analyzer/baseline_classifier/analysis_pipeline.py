import logging
from .role_classifier import RoleSimilarityAnalyzer
from .industry_classifier import IndustrySimilarityAnalyzer
from ..config import AppConfig
from ..utils.embedding_processor import BatchEmbeddingProcessor

logger = logging.getLogger(__name__)

class AnalysisPipeline:
    def __init__(self, embedding_model, semantic_baselines, text_preprocessor, config: AppConfig):
        self.embedding_model = embedding_model
        self.semantic_baselines = semantic_baselines
        self.text_preprocessor = text_preprocessor
        self.config = config
        
        embedding_processor = BatchEmbeddingProcessor(embedding_model)
        
        self.role_analyzer = RoleSimilarityAnalyzer(
            embedding_processor=embedding_processor,
            semantic_baselines=self.semantic_baselines,
            text_preprocessor=self.text_preprocessor,
            thresholds=self.config.model_thresholds
        )
        
        self.industry_analyzer = IndustrySimilarityAnalyzer(
            embedding_processor=embedding_processor,
            baselines=self.semantic_baselines.get('industry', {}),
            preprocessor=self.text_preprocessor,
            threshold=self.config.model_thresholds
        )
        
    def analyze(self, job_data: dict) -> dict:
        """
        Runs the analysis pipeline on a single job.
        
        Args:
            job_data (dict): A dictionary containing job details like 'job_title', 
                             'effective_description' (for role classification),
                             'company_description' (for industry classification),
                             'industry', and 'company_name'.
                             
        Returns:
            dict: A dictionary with analysis results, including similarity scores.
        """
        job_title = job_data.get('job_title', '')
        job_description = job_data.get('effective_description', '')  # For role classification
        company_description = job_data.get('company_description', '')  # For industry classification
        job_industry = job_data.get('industry', '')
        company_name = job_data.get('company_name', '')
        
        role_similarities = self.role_analyzer.calculate_similarities(job_title, job_description)
        industry_similarities = self.industry_analyzer.calculate_similarities(job_industry, company_name, company_description)

        # Sort similarities for easier consumption
        sorted_role_similarities = dict(sorted(role_similarities.items(), key=lambda item: item[1], reverse=True))
        sorted_industry_similarities = dict(sorted(industry_similarities.items(), key=lambda item: item[1], reverse=True))

        # For backward compatibility, we can still provide a "primary" classification.
        primary_role = "general"
        if sorted_role_similarities:
            best_role, best_score = next(iter(sorted_role_similarities.items()))
            if best_score >= self.config.model_thresholds.role_similarity_threshold:
                primary_role = best_role

        primary_industry = "Unknown"
        if sorted_industry_similarities:
            best_industry, best_score = next(iter(sorted_industry_similarities.items()))
            if best_score >= self.config.model_thresholds.industry_similarity_threshold:
                primary_industry = best_industry
        
        logger.info(f"Analysis complete. Primary Role='{primary_role}', Primary Industry='{primary_industry}'")
        
        return {
            'role': primary_role,
            'industry': primary_industry if primary_industry != "Unknown" else job_industry,
            'role_similarity_analysis': sorted_role_similarities,
            'industry_similarity_analysis': sorted_industry_similarities
        } 