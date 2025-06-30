import logging
import torch
from sentence_transformers import util
from ..config import ModelThresholds, PipelineConfig, CacheConfig
from ..utils.embedding_processor import BatchEmbeddingProcessor

logger = logging.getLogger(__name__)


class SemanticAnomalyDetector:
    def __init__(self, embedding_model, semantic_baselines, boilerplate_baseline, text_preprocessor, nlp_model, thresholds: ModelThresholds, pipeline_config: PipelineConfig, cache_config: CacheConfig):
        self.embedding_model = embedding_model
        self.semantic_baselines = semantic_baselines
        self.boilerplate_baseline = boilerplate_baseline
        self.text_preprocessor = text_preprocessor
        self.nlp = nlp_model
        self.thresholds = thresholds
        
        self.embedding_processor = BatchEmbeddingProcessor(
            embedding_model=self.embedding_model,
            batch_size=pipeline_config.batch_size,
            cache_size=cache_config.cache_size
        )
        # self.explanation_templates = {
        #     "Industry-Specific": {
        #         "explanation": "The requirement for '{skill}' is highly specific to the {industry} sector. While unusual for a typical {role} role, it signals a demand for deep industry knowledge.",
        #         "business_impact": "This is a key differentiator. Highlighting your understanding of '{skill}' and its application in the {industry} context can give you a significant advantage."
        #     },
        #     "Cross-Role": {
        #         "explanation": "Mentioning '{skill}' is uncommon for a {role} position. This could indicate the role is hybrid, involves collaboration with other teams, or that the company uses a unique tech stack.",
        #         "business_impact": "If you possess this skill, it's a valuable talking point that shows your versatility. If not, it's worth asking about to understand the team's structure and expectations."
        #     },
        #     "Emerging Tech": {
        #         "explanation": "The term '{skill}' represents a new or emerging technology. Its inclusion suggests the company is focused on innovation and exploring cutting-edge solutions.",
        #         "business_impact": "This can be an excellent opportunity for growth and learning. Be prepared to discuss your ability to adapt to new technologies and learn quickly."
        #     }
        # }

    def _create_hybrid_baseline(self, role_similarities: dict, top_k: int = 3):
        """Creates a weighted hybrid baseline from the top-k most similar roles."""
        if not role_similarities:
            return None, {}

        # Sort roles by similarity score and take the top-k
        sorted_roles = sorted(role_similarities.items(), key=lambda item: item[1], reverse=True)[:top_k]
        
        # Filter out roles with scores below a minimum threshold to avoid noise
        top_roles = {role: score for role, score in sorted_roles if score > self.thresholds.role_similarity_threshold}
        
        if not top_roles:
            return None, {}

        # Normalize the scores to use as weights
        total_score = sum(top_roles.values())
        weights = {role: score / total_score for role, score in top_roles.items()}

        # Combine the baseline vectors using the calculated weights
        combined_vectors = []
        for role, weight in weights.items():
            role_baseline = self.semantic_baselines.get('role', {}).get(role, {})
            vectors = role_baseline.get('vectors')
            if vectors is not None and vectors.numel() > 0:
                # In the new architecture, vectors is a 1D tensor. We collect them for stacking.
                combined_vectors.append(vectors)

        if not combined_vectors:
            return None, {}
        
        # Stack the vectors to create a (k, D) tensor, where k is the number of top roles.
        hybrid_baseline = torch.stack(combined_vectors, dim=0)
        
        logger.info(f"Created hybrid baseline from roles: {list(weights.keys())} with weights {list(weights.values())}")
        
        return hybrid_baseline, weights

    def detect_anomalies(self, target_job: str, analysis_result: dict):
        """
        Detects semantic anomalies using a more robust, context-aware approach.
        1. Filters out chunks that are core to the primary role.
        2. Identifies "cross-role" skills by checking against other role baselines.
        3. Filters out common boilerplate text using a negative baseline.
        """
        role_similarities = analysis_result.get('role_similarity_analysis', {})
        primary_role = analysis_result.get('role', 'general')

        if primary_role == 'general' or not role_similarities:
            logger.info("Primary role is 'general' or no similarities found, skipping semantic anomaly detection.")
            return {"semantic_anomalies": [], "baseline_composition": {}}

        # Create a hybrid baseline from the top roles for the primary role context
        primary_role_baseline, baseline_composition = self._create_hybrid_baseline(role_similarities)
        
        if primary_role_baseline is None or primary_role_baseline.numel() == 0:
            logger.warning(f"Could not create a valid primary role baseline. Cannot perform anomaly detection.")
            return {"semantic_anomalies": [], "baseline_composition": {}}

        # Prepare other role baselines for cross-checking
        other_role_baselines = {
            name: baseline['vectors']
            for name, baseline in self.semantic_baselines.get('role', {}).items()
            if name != primary_role and baseline.get('vectors') is not None
        }

        # --- Start of the new detection logic ---
        chunks = self.text_preprocessor.improved_chunking_for_anomaly_detection(target_job)
        if not chunks:
            return {"semantic_anomalies": [], "baseline_composition": {}}

        chunk_vectors = self.embedding_processor.encode_chunks(chunks)
        if chunk_vectors.numel() == 0:
            return {"semantic_anomalies": [], "baseline_composition": {}}

        anomalies = []
        for i, chunk in enumerate(chunks):
            chunk_vector = chunk_vectors[i].unsqueeze(0)

            # Step 1: Filter out core responsibilities and boilerplate
            sim_to_primary = self._calculate_max_similarity(chunk_vector, primary_role_baseline)
            sim_to_boilerplate = self._calculate_max_similarity(chunk_vector, self.boilerplate_baseline)

            if sim_to_primary > self.thresholds.core_skill_threshold:
                continue # It's a core skill, not an anomaly.

            if sim_to_boilerplate > self.thresholds.boilerplate_threshold:
                continue # It's standard boilerplate, not an anomaly.

            # Step 2: Identify "cross-role" anomalies
            cross_role_matches = []
            for role_name, other_baseline in other_role_baselines.items():
                sim_to_other = self._calculate_max_similarity(chunk_vector, other_baseline)
                if sim_to_other > self.thresholds.cross_role_threshold:
                    cross_role_matches.append({"role": role_name, "similarity": sim_to_other})
            
            # If the chunk is highly similar to another role, it's a valuable cross-role anomaly
            if cross_role_matches:
                # Find the best match among the cross-roles
                best_cross_match = max(cross_role_matches, key=lambda x: x['similarity'])
                anomalies.append({
                    "chunk": chunk,
                    "type": "Cross-Role",
                    "similarity_to_primary_role": round(sim_to_primary, 3),
                    "related_to_role": best_cross_match['role'],
                    "related_role_similarity": round(best_cross_match['similarity'], 3)
                })

        if not anomalies:
            logger.info("No semantic anomalies detected after contextual filtering.")
        else:
            logger.info(f"Detected {len(anomalies)} high-confidence semantic anomalies.")

        return {"semantic_anomalies": anomalies, "baseline_composition": baseline_composition}

    def _get_global_baseline_vectors(self):
        global_baselines = self.semantic_baselines.get('global', {})
        if global_baselines:
            global_baseline_name = next(iter(global_baselines))
            return global_baselines[global_baseline_name].get('vectors')
        return None

    def _calculate_max_similarity(self, chunk_vector, baseline_vectors):
        if baseline_vectors is None or baseline_vectors.numel() == 0:
            return 0.0
        # Ensure baseline_vectors is 2D
        if baseline_vectors.dim() == 1:
            baseline_vectors = baseline_vectors.unsqueeze(0)
        return torch.max(util.cos_sim(chunk_vector, baseline_vectors)).item() 