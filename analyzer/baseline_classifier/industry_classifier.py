import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ..utils.embedding_processor import BatchEmbeddingProcessor
from ..config import ModelThresholds
import torch

logger = logging.getLogger(__name__)

class IndustryClassifier:
    """
    Classifies job industries based on semantic similarity to predefined baselines.
    """
    def __init__(self, embedding_processor: BatchEmbeddingProcessor, baselines, preprocessor, threshold:ModelThresholds):
        """
        Initializes the IndustryClassifier.

        Args:
            embedding_processor: An instance of BatchEmbeddingProcessor.
            baselines (dict): A dictionary where keys are industry names and values are dictionaries
                              of baseline texts.
            preprocessor: An instance of TextPreprocessor.
            threshold (float): The similarity threshold for a successful classification.
        """
        self.embedding_processor = embedding_processor
        self.preprocessor = preprocessor
        self.threshold = threshold.industry_similarity_threshold
        self.baseline_embeddings = self._load_precomputed_vectors(baselines)

    def _load_precomputed_vectors(self, baselines):
        """
        Loads pre-computed vectors from the baselines.
        Averages the vectors for each baseline to create a single representative vector.
        """
        baseline_embeddings = {}
        if not baselines:
            logger.warning("No industry baselines provided to IndustryClassifier.")
            return baseline_embeddings

        logger.info(f"Loading pre-computed vectors for {len(baselines)} industries.")
        for industry_name, baseline_content in baselines.items():
            if not isinstance(baseline_content, dict):
                logger.warning(f"Skipping baseline for industry '{industry_name}' due to unexpected format: {type(baseline_content)}")
                continue

            vectors = baseline_content.get('vectors')
            if vectors is None or len(vectors) == 0:
                logger.warning(f"No vectors found for industry baseline '{industry_name}'.")
                continue

            try:
                # Create a single representative vector by averaging all skill vectors
                avg_vector = np.mean(np.array(vectors.cpu()), axis=0)
                baseline_embeddings[industry_name] = avg_vector.reshape(1, -1)
            except Exception as e:
                logger.error(f"Failed to process vectors for industry '{industry_name}': {e}", exc_info=True)
        
        logger.info(f"Successfully loaded and processed embeddings for {len(baseline_embeddings)} industry baselines.")
        return baseline_embeddings

    def classify(self, job_industry, company_name, job_description):
        """
        Classifies the industry of a job using a weighted scoring model.

        Args:
            job_industry (str): The industry of the job.
            company_name (str): The name of the company.
            job_description (str): The description of the job.

        Returns:
            str: The name of the classified industry, or "Unknown" if no suitable industry is found.
        """
        if not self.baseline_embeddings:
            logger.warning("Cannot classify industry: no baseline embeddings available.")
            return "Unknown"

        # Calculate scores from different parts of the job data
        industry_scores = self._calculate_industry_name_similarity(job_industry)
        desc_scores = self._calculate_description_similarity(f"{company_name} {job_description}")

        if not industry_scores and not desc_scores:
            logger.warning("No valid scores could be computed from industry or description.")
            return "Unknown"
        
        # Combine scores with a heavier weight on the industry name
        final_scores = self._weigh_and_combine_scores(industry_scores, desc_scores, industry_weight=0.7)

        if not final_scores:
            logger.info("No final scores after weighting. Classified as 'Unknown'.")
            return "Unknown"

        best_match_industry, max_similarity = max(final_scores.items(), key=lambda item: item[1])
        
        logger.debug(f"Industry classification best match: '{best_match_industry}' with score {max_similarity:.4f} (Threshold: {self.threshold})")

        if max_similarity >= self.threshold:
            logger.info(f"Final industry classification: '{best_match_industry}' with score {max_similarity:.4f}")
            return best_match_industry
        else:
            logger.info(f"Best match '{best_match_industry}' score {max_similarity:.4f} is below threshold {self.threshold}. Classified as 'Unknown'.")
            return "Unknown"

    def _weigh_and_combine_scores(self, industry_scores: dict, desc_scores: dict, industry_weight: float) -> dict:
        """Combines industry and description scores using a weighted average."""
        all_industries = set(industry_scores.keys()) | set(desc_scores.keys())
        final_scores = {}
        desc_weight = 1.0 - industry_weight

        logger.debug(f"Combining scores with industry_weight={industry_weight:.2f}")
        for industry in all_industries:
            industry_score = industry_scores.get(industry, 0.0)
            desc_score = desc_scores.get(industry, 0.0)
            
            # Apply penalties similar to RoleClassifier for robustness
            if industry_score > 0 and desc_score == 0:
                weighted_score = industry_score * 0.8  # Penalize if only industry name is available
            elif industry_score == 0 and desc_score > 0:
                weighted_score = desc_score * 0.6  # Penalize more if only description is available
            else:
                weighted_score = (industry_weight * industry_score) + (desc_weight * desc_score)
            
            final_scores[industry] = weighted_score
            logger.debug(f"Industry '{industry}': industry_score={industry_score:.4f}, desc_score={desc_score:.4f} -> final_score={weighted_score:.4f}")
                
        return final_scores

    def _calculate_industry_name_similarity(self, job_industry: str) -> dict:
        """Calculates similarity scores based on the job industry name."""
        if not job_industry or not job_industry.strip() or not self.baseline_embeddings:
            return {}

        # The industry name itself is compared against the pre-computed average vectors.
        # This is a bit different from RoleClassifier's title comparison but fits the current structure.
        cleaned_industry = self.preprocessor.preprocess_text(job_industry)
        if not cleaned_industry:
            return {}
            
        try:
            input_embedding = self.embedding_processor.get_embedding(cleaned_industry, 'job_industry_name_classification')
            if input_embedding is None:
                return {}

            scores = {}
            input_embedding_np = input_embedding.cpu().numpy().reshape(1, -1)
            for name, baseline_embedding in self.baseline_embeddings.items():
                similarity = cosine_similarity(input_embedding_np, baseline_embedding)[0][0]
                scores[name] = similarity
            logger.debug(f"Industry name similarity scores: {scores}")
            return scores
        except Exception as e:
            logger.error(f"Industry name embedding/similarity failed: {e}", exc_info=True)
            return {}

    def _calculate_description_similarity(self, text: str) -> dict:
        """Calculates similarity scores based on the job description and company name."""
        if not text or not text.strip():
            return {}

        preprocessed_text = self.preprocessor.preprocess_text(text)
        if not preprocessed_text.strip():
            return {}

        try:
            # Reusing the logic from the original classify method for description part
            job_embedding = self.embedding_processor.get_embedding(preprocessed_text, 'job_description_classification')
            if job_embedding is None:
                return {}
            
            job_embedding_np = job_embedding.cpu().numpy().reshape(1, -1)
            scores = {}
            for name, baseline_embedding in self.baseline_embeddings.items():
                similarity = cosine_similarity(job_embedding_np, baseline_embedding)[0][0]
                scores[name] = similarity
            
            logger.debug(f"Description similarity scores: {scores}")
            return scores
        except Exception as e:
            logger.error(f"Description embedding/similarity failed: {e}", exc_info=True)
            return {} 