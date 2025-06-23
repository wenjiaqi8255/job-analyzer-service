import logging
import torch
from sentence_transformers import util

logger = logging.getLogger(__name__)


class SemanticAnomalyDetector:
    def __init__(self, embedding_model, semantic_baselines, text_preprocessor, nlp_model):
        self.embedding_model = embedding_model
        self.semantic_baselines = semantic_baselines
        self.text_preprocessor = text_preprocessor
        self.nlp = nlp_model
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

    def detect_semantic_anomalies(self, target_job: str, role_baseline_name: str, 
                                industry_baseline_name: str = None, similarity_threshold=0.5):
        """
        检测语义异常，现在会自动过滤无关内容
        """
        if role_baseline_name == 'general':
            logger.info("Job role classified as 'general', skipping semantic anomaly detection.")
            return []

        if not self.embedding_model:
            logger.error("Embedding model is not available. Cannot perform semantic analysis.")
            return []
        
        # 使用新的过滤方法
        chunks = self.text_preprocessor.chunk_and_filter_text(target_job)
        
        if not chunks:
            logger.warning("No relevant chunks after content filtering")
            return []
        
        logger.info(f"Processing {len(chunks)} filtered chunks for anomaly detection")        

        chunk_vectors = self.embedding_model.encode(chunks, convert_to_tensor=True)
        
        role_baseline_vectors = self.semantic_baselines.get('role', {}).get(role_baseline_name, {}).get("vectors")
        if role_baseline_vectors is None:
            logger.warning(f"Role baseline '{role_baseline_name}' has no vectors.")
            return []

        industry_baseline_vectors = self.semantic_baselines.get("industry", {}).get(industry_baseline_name, {}).get("vectors")
        global_baseline_vectors = self._get_global_baseline_vectors()

        anomalies = []
        for i, chunk in enumerate(chunks):
            sim_role = self._calculate_max_similarity(chunk_vectors[i].unsqueeze(0), role_baseline_vectors)
            
            if sim_role < similarity_threshold:
                anomaly = self._build_anomaly_record(
                    chunk, chunk_vectors[i], sim_role, role_baseline_name, industry_baseline_name, 
                    role_baseline_vectors, industry_baseline_vectors, global_baseline_vectors
                )
                anomalies.append(anomaly)
        
        return anomalies

    def _get_global_baseline_vectors(self):
        global_baselines = self.semantic_baselines.get('global', {})
        if global_baselines:
            global_baseline_name = next(iter(global_baselines))
            return global_baselines[global_baseline_name].get('vectors')
        return None

    def _build_anomaly_record(self, chunk, chunk_vector, sim_role, role_name, industry_name, role_vectors, industry_vectors, global_vectors):
        sim_industry = self._calculate_max_similarity(chunk_vector, industry_vectors)
        sim_global = self._calculate_max_similarity(chunk_vector, global_vectors)
        
        anomaly_type = "Cross-Role"
        if sim_global < 0.35:
            anomaly_type = "Emerging Tech"
        elif industry_name and sim_industry > 0.65 and sim_role < 0.4:
            anomaly_type = "Industry-Specific"

        skill_topic = "unknown"
        if anomaly_type == "Industry-Specific" and industry_vectors is not None:
            keywords = self.semantic_baselines.get("industry", {}).get(industry_name, {}).get("keywords", [])
            skill_topic = self._find_best_match_keyword(chunk, chunk_vector, industry_vectors, keywords)
        else:
            keywords = self.semantic_baselines.get("role", {}).get(role_name, {}).get("keywords", [])
            skill_topic = self._find_best_match_keyword(chunk, chunk_vector, role_vectors, keywords)

        # template = self.explanation_templates.get(anomaly_type, {})
        # explanation = template.get("explanation", "").format(skill=skill_topic, industry=industry_name, role=role_name)
        # business_impact = template.get("business_impact", "").format(skill=skill_topic, industry=industry_name, role=role_name)

        return {
            "chunk": chunk, "type": anomaly_type, 
            # "explanation": explanation, 
            # "business_impact": business_impact,
            "similarity_to_role": round(sim_role, 3), "similarity_to_industry": round(sim_industry, 3), "similarity_to_global": round(sim_global, 3),
        }

    def _find_best_match_keyword(self, chunk_text, chunk_vector, baseline_vectors, baseline_keywords):
        if baseline_vectors is None or len(baseline_vectors) == 0 or not baseline_keywords:
            doc = self.nlp(chunk_text)
            for np in doc.noun_chunks:
                return np.text
            return "this requirement"

        similarities = util.cos_sim(chunk_vector, baseline_vectors)
        best_match_index = torch.argmax(similarities).item()
        return baseline_keywords[best_match_index]

    def _calculate_max_similarity(self, chunk_vector, baseline_vectors):
        if baseline_vectors is None or len(baseline_vectors) == 0:
            return 0.0
        return torch.max(util.cos_sim(chunk_vector, baseline_vectors)).item() 