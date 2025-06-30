import logging
import torch
from sentence_transformers import util
from ..config import ModelThresholds
from ..utils.embedding_processor import BatchEmbeddingProcessor

logger = logging.getLogger(__name__)


class RoleSimilarityAnalyzer:
    def __init__(self, embedding_processor: BatchEmbeddingProcessor, semantic_baselines, text_preprocessor, thresholds: ModelThresholds):
        self.embedding_processor = embedding_processor
        self.semantic_baselines = semantic_baselines
        self.text_preprocessor = text_preprocessor
        self.thresholds = thresholds
        self.role_title_embeddings = {}
        self._precompute_role_title_embeddings()

    def _precompute_role_title_embeddings(self):
        """Pre-computes and caches embeddings for all standard role titles."""
        logger.info("Pre-computing role title embeddings...")
        role_baselines = self.semantic_baselines.get("role", {})
        if not role_baselines:
            logger.warning("No role baselines available for title embedding precomputation.")
            return

        role_titles_to_encode = [self._role_key_to_title(name.replace('_baseline', '')) for name in role_baselines.keys()]
        role_names = list(role_baselines.keys())

        try:
            title_embeddings = self.embedding_processor.get_embeddings(role_titles_to_encode, 'role_title_precomputation')
            self.role_title_embeddings = dict(zip(role_names, title_embeddings))
            logger.info(f"✅ Successfully pre-computed {len(self.role_title_embeddings)} role title embeddings.")
            logger.debug(f"Cached role keys: {list(self.role_title_embeddings.keys())}")
        except Exception as e:
            logger.error(f"Failed to pre-compute role title embeddings: {e}")
            self.role_title_embeddings = {}

    def _role_key_to_title(self, role_key: str) -> str:
        """Converts a role key into a natural language title."""
        title = role_key.replace('_', ' ').title()
        replacements = {'Ui Ux': 'UI UX', 'Ai Ml': 'AI ML', 'It': 'IT', 'Seo': 'SEO', 'Api': 'API', 'Devops': 'DevOps', 'Qa': 'QA'}
        for old, new in replacements.items():
            title = title.replace(old, new)
        return title

    def calculate_similarities(self, job_title: str = "", job_description: str = "") -> dict:
        if not self.semantic_baselines.get("role"):
            logger.warning("Cannot calculate role similarities due to missing baselines.")
            return {}
        
        logger.debug(f"Calculating role similarities for title='{job_title}' and description='{job_description[:50]}...'")

        title_scores = self._calculate_title_similarity(job_title)
        desc_scores = self._calculate_description_similarity(job_description)

        if not title_scores and not desc_scores:
            logger.debug("No valid title or description scores computed.")
            return {}

        final_scores = self._weigh_and_combine_scores(title_scores, desc_scores, self.thresholds.role_title_weight)
        
        if not final_scores:
            return {}

        logger.info(f"Calculated {len(final_scores)} role similarity scores.")
        return final_scores

    def _calculate_title_similarity(self, job_title: str) -> dict:
        """Calculates similarity scores based on the job title."""
        if not job_title or not job_title.strip() or not self.role_title_embeddings:
            return {}
        
        cleaned_title = self.text_preprocessor.preprocess_job_title(job_title)
        logger.debug(f"Cleaned title for similarity calculation: '{cleaned_title}'")
        if not cleaned_title:
            return {}
            
        try:
            input_embedding = self.embedding_processor.get_embedding(cleaned_title, 'job_title_classification')
            if input_embedding is None:
                return {}
            scores = {}
            for name, cached_embedding in self.role_title_embeddings.items():
                similarity = util.cos_sim(input_embedding.unsqueeze(0), cached_embedding.unsqueeze(0)).item()
                scores[name] = similarity
            logger.debug(f"Title similarity scores: {scores}")
            return scores
        except Exception as e:
            logger.error(f"Title embedding failed: {e}")
            return {}

    def _calculate_description_similarity(self, job_description: str) -> dict:
        """Calculates similarity scores based on the job description."""
        if not job_description or not job_description.strip():
            return {}

        chunks = self.text_preprocessor.chunk_text(job_description)
        logger.debug(f"Processing {len(chunks)} description chunks for similarity.")
        if not chunks:
            return {}

        try:
            chunk_vectors = self.embedding_processor.encode_chunks(chunks)
            scores = {}
            role_baselines = self.semantic_baselines.get("role", {})
            for name, baseline in role_baselines.items():
                baseline_vectors = baseline.get('vectors')
                if baseline_vectors is None or len(baseline_vectors) == 0:
                    scores[name] = 0.0
                    continue

                chunk_max_sims = [torch.max(util.cos_sim(chunk_vec.unsqueeze(0), baseline_vectors)).item() for chunk_vec in chunk_vectors]
                scores[name] = sum(chunk_max_sims) / len(chunk_max_sims)
            logger.debug(f"Description similarity scores: {scores}")
            return scores
        except Exception as e:
            logger.error(f"Description embedding failed: {e}")
            return {}

    def _weigh_and_combine_scores(self, title_scores: dict, desc_scores: dict, title_weight: float) -> dict:
        """Combines title and description scores using a weighted average."""
        all_roles = set(title_scores.keys()) | set(desc_scores.keys())
        final_scores = {}
        desc_weight = 1.0 - title_weight

        logger.debug(f"Combining scores with title_weight={title_weight:.2f}")
        for role in all_roles:
            title_score = title_scores.get(role, 0.0)
            desc_score = desc_scores.get(role, 0.0)
            
            weighted_score = 0.0
            if title_score > 0 and desc_score == 0:
                weighted_score = title_score * 0.8  # Penalize if only title is available
            elif title_score == 0 and desc_score > 0:
                weighted_score = desc_score * 0.6  # Penalize more if only description is available
            else:
                weighted_score = (title_weight * title_score) + (desc_weight * desc_score)
            
            final_scores[role] = weighted_score
            logger.debug(f"Role '{role}': title_score={title_score:.4f}, desc_score={desc_score:.4f} -> final_score={weighted_score:.4f}")
                
        return final_scores 