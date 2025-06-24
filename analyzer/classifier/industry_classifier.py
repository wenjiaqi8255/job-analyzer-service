import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ..embedding_processor import BatchEmbeddingProcessor

logger = logging.getLogger(__name__)

class IndustryClassifier:
    """
    Classifies job industries based on semantic similarity to predefined baselines.
    """
    def __init__(self, embedding_model, baselines, preprocessor, threshold=0.5):
        """
        Initializes the IndustryClassifier.

        Args:
            embedding_model: The model to use for generating embeddings.
            baselines (dict): A dictionary where keys are industry names and values are dictionaries
                              of baseline texts.
            preprocessor: An instance of TextPreprocessor.
            threshold (float): The similarity threshold for a successful classification.
        """
        self.embedding_model = embedding_model
        self.preprocessor = preprocessor
        self.threshold = threshold
        self.embedding_processor = BatchEmbeddingProcessor(embedding_model)
        self.baseline_embeddings = self._compute_baseline_embeddings(baselines)

    def _compute_baseline_embeddings(self, baselines):
        """
        Computes and caches embeddings for the baseline industries.

        Args:
            baselines (dict): The baseline industry data.

        Returns:
            dict: A dictionary mapping industry names to their computed embeddings.
        """
        baseline_embeddings = {}
        if not baselines:
            logger.warning("No industry baselines provided to IndustryClassifier.")
            return baseline_embeddings
            
        logger.info(f"Computing baseline embeddings for {len(baselines)} industries.")
        for industry_name, baseline_content in baselines.items():
            if not isinstance(baseline_content, dict):
                logger.warning(f"Skipping baseline for industry '{industry_name}' due to unexpected format: {type(baseline_content)}")
                continue

            texts_to_embed = baseline_content.get('texts', [])
            if not texts_to_embed and isinstance(baseline_content, dict):
                 texts_to_embed = [text for key, text in baseline_content.items() if isinstance(text, str)]

            if not texts_to_embed:
                logger.warning(f"No text found to create embedding for industry baseline '{industry_name}'.")
                continue
            
            combined_text = " ".join(self.preprocessor.preprocess_texts(texts_to_embed))
            
            if not combined_text.strip():
                logger.warning(f"No content to embed for industry '{industry_name}' after preprocessing.")
                continue

            try:
                embedding = self.embedding_processor.get_embedding(combined_text, 'industry_baseline')
                if embedding is not None:
                    baseline_embeddings[industry_name] = np.array(embedding).reshape(1, -1)
            except Exception as e:
                logger.error(f"Failed to compute embedding for industry baseline '{industry_name}': {e}", exc_info=True)
                
        logger.info(f"Successfully computed embeddings for {len(baseline_embeddings)} industry baselines.")
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
        preprocessed_text = self.preprocessor.preprocess_text(job_text)
        
        if not preprocessed_text.strip():
            logger.warning("Cannot classify industry: no content after preprocessing.")
            return "Unknown"

        try:
            job_embedding = self.embedding_processor.get_embedding(preprocessed_text, 'job_industry_classification')
            if job_embedding is None:
                return "Unknown"
            job_embedding = np.array(job_embedding).reshape(1, -1)

            max_similarity = -1
            best_match_industry = "Unknown"

            for industry_name, baseline_embedding in self.baseline_embeddings.items():
                similarity = cosine_similarity(job_embedding, baseline_embedding)[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_industry = industry_name
            
            logger.debug(f"Industry classification for '{job_industry}': Best match is '{best_match_industry}' with similarity {max_similarity:.4f}")

            if max_similarity >= self.threshold:
                return best_match_industry
            else:
                return "Unknown"
        except Exception as e:
            logger.error(f"Error during industry classification for job industry '{job_industry}': {e}", exc_info=True)
            return "Unknown" 