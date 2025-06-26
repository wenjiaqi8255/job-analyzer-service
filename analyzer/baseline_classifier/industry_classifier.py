import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ..utils.embedding_processor import BatchEmbeddingProcessor
import torch

logger = logging.getLogger(__name__)

class IndustryClassifier:
    """
    Classifies job industries based on semantic similarity to predefined baselines.
    """
    def __init__(self, embedding_processor: BatchEmbeddingProcessor, baselines, preprocessor, threshold=0.5):
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
        self.threshold = threshold
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
        Classifies the industry of a job.

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

        job_text = f"{job_industry} {company_name} {job_description}"
        logger.debug(f"Classifying industry based on text: '{job_text[:100]}...'")
        preprocessed_text = self.preprocessor.preprocess_text(job_text)
        
        if not preprocessed_text.strip():
            logger.warning("Cannot classify industry: no content after preprocessing.")
            return "Unknown"

        try:
            job_embedding = self.embedding_processor.get_embedding(preprocessed_text, 'job_industry_classification')
            if job_embedding is None:
                return "Unknown"
            # job_embedding is a tensor, needs to be reshaped for cosine_similarity
            job_embedding_np = job_embedding.cpu().numpy().reshape(1, -1)

            max_similarity = -1
            best_match_industry = "Unknown"

            for industry_name, baseline_embedding in self.baseline_embeddings.items():
                similarity = cosine_similarity(job_embedding_np, baseline_embedding)[0][0]
                logger.debug(f"  - Similarity with '{industry_name}': {similarity:.4f}")
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_industry = industry_name
            
            logger.debug(f"Industry classification best match: '{best_match_industry}' with similarity {max_similarity:.4f} (Threshold: {self.threshold})")

            if max_similarity >= self.threshold:
                logger.info(f"Final industry classification: '{best_match_industry}'")
                return best_match_industry
            else:
                logger.info(f"Industry similarity below threshold. Classified as 'Unknown'.")
                return "Unknown"
        except Exception as e:
            logger.error(f"Error during industry classification for job industry '{job_industry}': {e}", exc_info=True)
            return "Unknown" 