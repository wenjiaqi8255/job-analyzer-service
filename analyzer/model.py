import re
import pandas as pd
import numpy as np
import spacy
import json
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from extractor.industry_keyword_library import INDUSTRY_KEYWORD_LIBRARY
from sentence_transformers import SentenceTransformer, util
import torch
from supabase import Client
from .db_queries import fetch_active_semantic_baselines, fetch_baseline_vectors

import logging
logger = logging.getLogger(__name__)

class EnhancedJobAnomalyDetector:
    def __init__(self, supabase_client: Client = None):
        self.supabase_client = supabase_client
        try:
            self.nlp = spacy.load("de_core_news_sm")
            logger.info("Loaded German spaCy model.")
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.warning("German model not found, using English model.")
            except OSError:
                logger.error("Please install spaCy model: python -m spacy download de_core_news_sm")
                raise
        
        self.embedding_model_name = 'all-MiniLM-L6-v2'
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded sentence-transformer model '{self.embedding_model_name}'.")
        except Exception as e:
            logger.error(f"Failed to load sentence-transformer model: {e}", exc_info=True)
            self.embedding_model = None

        # Determine the device for tensor operations. This will be used to ensure
        # all tensors are on the same device as the model to prevent runtime errors.
        if self.embedding_model:
            self.device = self.embedding_model.device
            logger.info(f"Using device: {self.device} for tensor operations.")
        else:
            self.device = "cpu"
            logger.warning("Embedding model not loaded. Defaulting to CPU for tensor operations.")

        self.semantic_baselines = {}
        if self.supabase_client and self.embedding_model:
            self._load_baselines_from_db()

        self.explanation_templates = {
            "Industry-Specific": {
                "explanation": "The requirement for '{skill}' is highly specific to the {industry} sector. While unusual for a typical {role} role, it signals a demand for deep industry knowledge.",
                "business_impact": "This is a key differentiator. Highlighting your understanding of '{skill}' and its application in the {industry} context can give you a significant advantage."
            },
            "Cross-Role": {
                "explanation": "Mentioning '{skill}' is uncommon for a {role} position. This could indicate the role is hybrid, involves collaboration with other teams, or that the company uses a unique tech stack.",
                "business_impact": "If you possess this skill, it's a valuable talking point that shows your versatility. If not, it's worth asking about to understand the team's structure and expectations."
            },
            "Emerging Tech": {
                "explanation": "The term '{skill}' represents a new or emerging technology. Its inclusion suggests the company is focused on innovation and exploring cutting-edge solutions.",
                "business_impact": "This can be an excellent opportunity for growth and learning. Be prepared to discuss your ability to adapt to new technologies and learn quickly."
            }
        }
        
        self.stop_words = {
            # 德语停用词
            'und', 'der', 'die', 'das', 'für', 'von', 'mit', 'bei', 'den', 'dem', 'des', 
            'ein', 'eine', 'einen', 'einer', 'sich', 'wir', 'sie', 'ihr', 'ich', 'du',
            'ist', 'sind', 'war', 'waren', 'haben', 'hat', 'wird', 'werden', 'kann',
            'soll', 'sollte', 'muss', 'auf', 'zu', 'nach', 'über', 'unter', 'durch',
            
            # 英语停用词
            'the', 'and', 'for', 'with', 'are', 'you', 'will', 'can', 'have', 'your', 
            'our', 'this', 'that', 'work', 'working', 'job', 'position', 'role',
            'also', 'access', 'employees', 'company', 'team', 'opportunity',
            'during', 'right', 'therefore', 'should', 'needed', 'responsible',
            'something', 'experience', 'skills', 'requirements', 'candidates',
            
            # 招聘领域通用词
            'experience', 'skills', 'requirements', 'candidate', 'candidates',
            'position', 'role', 'job', 'work', 'working', 'opportunity',
            'team', 'company', 'employees', 'department', 'organization',
            'responsibilities', 'tasks', 'duties', 'qualifications',
            'background', 'knowledge', 'ability', 'abilities', 'capable',
            
            # 德语招聘通用词
            'stelle', 'position', 'arbeitsplatz', 'mitarbeiter', 'team',
            'unternehmen', 'firma', 'aufgaben', 'anforderungen', 'qualifikationen',
            'erfahrung', 'kenntnisse', 'fähigkeiten', 'verantwortung',
        }
        self.company_suffixes = {
            'gmbh', 'ag', 'kg', 'ohg', 'gbr', 'ug', 'eg', 'ev',
            'inc', 'corp', 'ltd', 'llc', 'co', 'company', 'group',
            'holding', 'ventures', 'capital', 'partners', 'solutions',
            'technologies', 'systems', 'services', 'consulting',
            'verlag', 'verlagsgruppe', 'media', 'publishing'
        }
        self.noise_patterns = [
            r'\w*_date_?\d*',     # publication_date_10
            r'job_id_?\d*',       # job_id_123  
            r'location_\w+',      # location_munich
            r'\d{4,}',            # 长数字ID
            r'publication_date',   # 明确过滤
            r'legal_entity',      # 法律实体
            r'^.{1,2}$',         # 1-2字符的词
            r'^\d+$',            # 纯数字
            r'^[a-z]$',          # 单字母
            r'http[s]?://.*',    # URL
            r'.*@.*\..*',        # 邮箱
        ]
        self.meaningful_pos = {
            'NOUN', 'PROPN',     # 名词、专有名词
            'ADJ',               # 形容词  
            'VERB',              # 动词
            'NUM'                # 数字
        }
        self.global_idf_cache = None
        self.industry_cache = {}

    def _load_baselines_from_db(self):
        """
        Loads active baselines and their pre-computed vectors from the database.
        (FINAL FIX - Handles JSON string parsing)
        """
        logger.info("Loading semantic baselines from database...")
        baselines = fetch_active_semantic_baselines(self.supabase_client)
        if not baselines:
            logger.warning("No baselines loaded from DB. Semantic analysis will be disabled.")
            return

        self.semantic_baselines = {"role": {}, "industry": {}, "global": {}}

        for baseline in baselines:
            baseline_id = baseline['id']
            name = baseline['name']
            baseline_type = baseline['baseline_type']
            data = baseline.get('baseline_data', {})

            # 1. 提取关键词
            source_keywords = []
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        source_keywords.extend(value)
            
            if not source_keywords:
                logger.warning(f"[{name}] No source keywords found. Skipping.")
                continue

            # 2. 获取向量数据
            vectors_data = fetch_baseline_vectors(self.supabase_client, baseline_id, self.embedding_model_name)
            if not vectors_data:
                logger.error(f"[{name}] CRITICAL: No vector data found in baseline_vectors table!")
                continue

            # 3. *** THE FINAL FIX IS HERE ***
            # 检查返回的是否是字符串，如果是，则解析为字典
            vectors_dict = {}
            if isinstance(vectors_data, str):
                try:
                    vectors_dict = json.loads(vectors_data)
                    logger.info(f"[{name}] Successfully parsed JSON string into a dictionary with {len(vectors_dict)} keys.")
                except json.JSONDecodeError:
                    logger.error(f"[{name}] FATAL: Failed to parse the received string as JSON. String starts with: {vectors_data[:100]}")
                    continue
            elif isinstance(vectors_data, dict):
                vectors_dict = vectors_data # 如果已经是字典，直接使用
            else:
                logger.error(f"[{name}] FATAL: Received vector data is neither a string nor a dictionary. Type is {type(vectors_data)}. Skipping.")
                continue
            
            # 4. 匹配关键词和向量
            vector_keys = set(vectors_dict.keys())
            matched_keywords = [kw for kw in source_keywords if kw in vector_keys]

            if not matched_keywords:
                logger.error(f"[{name}] FATAL MISMATCH: 0 keywords matched! Keywords from semantic_baselines are MISSING in the vector dictionary.")
                # 为了避免刷屏，只打印少量示例
                logger.error(f"[{name}] Example keywords from semantic_baselines: {source_keywords[:5]}")
                logger.error(f"[{name}] Example keys from vector dictionary: {list(vector_keys)[:5]}")
                continue

            # 5. 基于匹配到的关键词构建 Tensor
            try:
                vectors_list = [vectors_dict[kw] for kw in matched_keywords]
                # Move tensor to the same device as the embedding model to prevent device mismatch errors
                vectors = torch.tensor(vectors_list, dtype=torch.float).to(self.device)

                logger.info(f"[{name}] Successfully created a tensor of shape {vectors.shape} on device '{self.device}' using {len(matched_keywords)} matched keywords.")

                if baseline_type in self.semantic_baselines:
                    self.semantic_baselines[baseline_type][name] = {
                        "keywords": matched_keywords,
                        "vectors": vectors
                    }
            except Exception as e:
                logger.error(f"[{name}] Error creating tensor: {e}", exc_info=True)

        logger.info("Finished loading baselines.")
        for b_type, b_dict in self.semantic_baselines.items():
            logger.info(f"Final loaded count for type '{b_type}': {len(b_dict)} baselines.")

    def classify_job_role(self, job_description: str, min_avg_similarity=0.35) -> str:
        """
        Classifies the job role by finding which role baseline has the highest
        average maximum similarity across all job description chunks.
        """
        if not self.embedding_model or not job_description:
            return "general"

        role_baselines = self.semantic_baselines.get("role", {})
        if not role_baselines:
            logger.warning("No role baselines loaded. Cannot classify role, defaulting to 'general'.")
            return "general"

        chunks = self.chunk_text(job_description)
        if not chunks:
            return "general"

        chunk_vectors = self.embedding_model.encode(chunks, convert_to_tensor=True)

        # 改为存储每个 role 的平均相似度分数
        role_avg_scores = {role_name: 0.0 for role_name in role_baselines.keys()}

        for role_name, baseline in role_baselines.items():
            baseline_vectors = baseline.get('vectors')
            if baseline_vectors is None or len(baseline_vectors) == 0:
                continue

            # 计算相似度矩阵 (这部分不变)
            similarities = util.cos_sim(chunk_vectors, baseline_vectors)
            # 找到每个 chunk 对这个 baseline 的最高相似度 (这部分不变)
            max_sim_per_chunk = torch.max(similarities, dim=1).values
            
            # 核心改动：计算这些最高相似度的平均值，而不是计算 "hits"
            average_score = torch.mean(max_sim_per_chunk).item()
            role_avg_scores[role_name] = average_score

        if not role_avg_scores:
            return "general"

        # 找到平均分最高的 role
        best_role = max(role_avg_scores, key=role_avg_scores.get)
        best_score = role_avg_scores[best_role]
        
        # 增加一个"安全阀"：如果最高的平均分也极低，说明完全不相关，还是返回 general
        if best_score < min_avg_similarity:
            logger.info(f"Best role '{best_role}' only scored {best_score:.3f} (below threshold {min_avg_similarity}). Classifying as 'general'. (Scores: { {k: round(v, 3) for k, v in role_avg_scores.items()} })")
            return "general"

        logger.info(f"Classified job role as '{best_role}' with score {best_score:.3f}. (All Scores: { {k: round(v, 3) for k, v in role_avg_scores.items()} })")
        return best_role

    def detect_semantic_anomalies(self, target_job: str, role_baseline_name: str, industry_baseline_name: str = None, similarity_threshold=0.5):
        """
        Detects semantic anomalies using vector similarity against pre-defined baselines.
        """
        # If the role is classified as general, we skip the analysis as there's no specific baseline for comparison.
        if role_baseline_name == 'general':
            logger.info("Job role classified as 'general', skipping semantic anomaly detection.")
            return []

        if not self.embedding_model:
            logger.error("Embedding model is not available. Cannot perform semantic analysis.")
            return []
        
        chunks = self.chunk_text(target_job)
        if not chunks:
            return []

        chunk_vectors = self.embedding_model.encode(chunks, convert_to_tensor=True)
        
        # --- Baseline Vector Retrieval ---
        role_baseline_vectors = self.semantic_baselines.get('role', {}).get(role_baseline_name, {}).get("vectors")
        if role_baseline_vectors is None:
            logger.warning(f"Role baseline '{role_baseline_name}' not found or has no vectors. Analysis cannot proceed.")
            return []

        industry_baseline_vectors = self.semantic_baselines.get("industry", {}).get(industry_baseline_name, {}).get("vectors")
        
        global_baselines = self.semantic_baselines.get('global', {})
        global_baseline_vectors = None
        if global_baselines:
            global_baseline_name = next(iter(global_baselines))
            global_baseline_vectors = global_baselines[global_baseline_name].get('vectors')

        # --- Anomaly Detection Loop ---
        anomalies = []
        for i, chunk in enumerate(chunks):
            chunk_vector = chunk_vectors[i].unsqueeze(0)
            
            sim_role = self._calculate_max_similarity(chunk_vector, role_baseline_vectors)
            
            if sim_role < similarity_threshold:
                sim_industry = self._calculate_max_similarity(chunk_vector, industry_baseline_vectors)
                sim_global = self._calculate_max_similarity(chunk_vector, global_baseline_vectors)
                
                anomaly_type = "Cross-Role"
                if sim_global < 0.35:
                    anomaly_type = "Emerging Tech"
                elif industry_baseline_name and sim_industry > 0.65 and sim_role < 0.4:
                    anomaly_type = "Industry-Specific"

                skill_topic = "unknown"
                if anomaly_type == "Industry-Specific" and industry_baseline_vectors is not None:
                    skill_topic = self._find_best_match_keyword(chunk, chunk_vector, industry_baseline_vectors, self.semantic_baselines.get("industry", {}).get(industry_baseline_name, {}).get("keywords", []))
                else:
                    skill_topic = self._find_best_match_keyword(chunk, chunk_vector, role_baseline_vectors, self.semantic_baselines.get("role", {}).get(role_baseline_name, {}).get("keywords", []))

                template = self.explanation_templates.get(anomaly_type, {})
                explanation = template.get("explanation", "").format(skill=skill_topic, industry=industry_baseline_name, role=role_baseline_name)
                business_impact = template.get("business_impact", "").format(skill=skill_topic, industry=industry_baseline_name, role=role_baseline_name)

                anomalies.append({
                    "chunk": chunk,
                    "type": anomaly_type,
                    "explanation": explanation,
                    "business_impact": business_impact,
                    "similarity_to_role": round(sim_role, 3),
                    "similarity_to_industry": round(sim_industry, 3),
                    "similarity_to_global": round(sim_global, 3),
                })
        
        return anomalies

    def _find_best_match_keyword(self, chunk_text, chunk_vector, baseline_vectors, baseline_keywords):
        """Finds the most similar keyword from a baseline to a given chunk vector."""
        if baseline_vectors is None or len(baseline_vectors) == 0 or not baseline_keywords:
            # As a fallback, try to extract a noun phrase from the chunk
            doc = self.nlp(chunk_text)
            for np in doc.noun_chunks:
                return np.text
            return "this requirement"

        similarities = util.cos_sim(chunk_vector, baseline_vectors)
        best_match_index = torch.argmax(similarities).item()
        return baseline_keywords[best_match_index]

    def _calculate_max_similarity(self, chunk_vector, baseline_vectors):
        """Helper to calculate max cosine similarity."""
        if baseline_vectors is None or len(baseline_vectors) == 0:
            return 0.0
        similarities = util.cos_sim(chunk_vector, baseline_vectors)
        return torch.max(similarities).item()

    def calculate_global_idf(self, descriptions: list[str]) -> dict:
        """
        Calculates global IDF values from a list of job descriptions.
        """
        logger.info("Calculating global IDF from descriptions...")

        # Filter out any empty documents
        all_texts = [text for text in descriptions if text and len(text.strip()) > 10]

        if len(all_texts) < 5:
            logger.warning("Not enough documents with keywords to calculate IDF, skipping.")
            return {}

        vectorizer = TfidfVectorizer(
            max_features=20000,
            stop_words=list(self.stop_words),
            ngram_range=(1, 2),
            min_df=3,
            token_pattern=r'(?u)\b[\w-]{2,}\b'  # Correct pattern for words with hyphens
        )
        try:
            vectorizer.fit(all_texts)
            feature_names = vectorizer.get_feature_names_out()
            idf_values = vectorizer.idf_
            idf_dict = dict(zip(feature_names, idf_values))
            self.global_idf_cache = idf_dict  # Still cache in memory for the current run
            
            logger.info(f"IDF calculation complete. Found {len(idf_dict)} terms.")
            sorted_idf = sorted(idf_dict.items(), key=lambda x: x[1])
            logger.info(f"Most common terms (low IDF): {[word for word, idf in sorted_idf[:10]]}")
            return idf_dict
        except Exception as e:
            logger.error(f"IDF calculation failed: {e}", exc_info=True)
            return {}

    def classify_job_industry(self, company_name, job_title, description):
        # ... existing code ...
        cache_key = f"{company_name}_{job_title}"
        if cache_key in self.industry_cache:
            return self.industry_cache[cache_key]
        combined_text = f"{company_name} {job_title} {description}".lower()
        industry_keywords = INDUSTRY_KEYWORD_LIBRARY
        industry_scores = {}
        for industry, keywords in industry_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                industry_scores[industry] = score
        if industry_scores:
            classified_industry = max(industry_scores, key=industry_scores.get)
        else:
            classified_industry = 'general'
        self.industry_cache[cache_key] = classified_industry
        return classified_industry

    def is_generic_word(self, word, idf_threshold=2.5):
        if not self.global_idf_cache:
            return False
        idf_value = self.global_idf_cache.get(word.lower(), float('inf'))
        return idf_value < idf_threshold

    def extract_company_terms(self, company_name):
        if not company_name or pd.isna(company_name):
            return set()
        company_terms = set()
        company_name = str(company_name).lower()
        try:
            doc = self.nlp(company_name)
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PERSON']:
                    words = re.findall(r'\b[a-zA-ZäöüÄÖÜß]{2,}\b', ent.text.lower())
                    company_terms.update(words)
        except:
            pass
        cleaned_name = re.sub(r'\([^)]*\)', '', company_name)
        cleaned_name = re.sub(r'[^\w\s\-]', ' ', cleaned_name)
        words = re.findall(r'\b[a-zA-ZäöüÄÖÜß]{2,}\b', cleaned_name)
        company_terms.update(words)
        filtered_terms = set()
        for word in company_terms:
            if word not in self.company_suffixes:
                filtered_terms.add(word)
            else:
                filtered_terms.add(word)
        return filtered_terms

    def chunk_text(self, text: str) -> list[str]:
        """
        Splits a job description into semantic chunks (sentences or meaningful blocks).
        This is the new preprocessing step for semantic analysis, replacing n-gram extraction.
        As per the new architecture, this prepares the text for vector-based comparison.
        """
        if not text or pd.isna(text):
            return []

        # Normalize text slightly first
        text = re.sub(r'\s+', ' ', text).strip()

        # Use spaCy's robust sentence segmentation
        doc = self.nlp(text)
        chunks = [sent.text.strip() for sent in doc.sents]

        # Further refine chunks: split by newlines which often delimit list items,
        # and filter out chunks that are too short to have semantic meaning.
        final_chunks = []
        for chunk in chunks:
            sub_chunks = chunk.split('\n')
            for sub_chunk in sub_chunks:
                sub_chunk = sub_chunk.strip()
                # A chunk should have at least a few words to be meaningful
                if sub_chunk and len(sub_chunk.split()) > 2:
                    final_chunks.append(sub_chunk)

        return final_chunks

    def advanced_text_preprocessing(self, text, company_terms_to_filter=None):
        if not text or pd.isna(text):
            return [], []
        if company_terms_to_filter is None:
            company_terms_to_filter = set()
        text = str(text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'[^\w\s\-/]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().lower()
        doc = self.nlp(text)
        unigrams = []
        bigrams = []
        valid_tokens = []
        for token in doc:
            if (token.is_punct or token.is_space or 
                token.text.lower() in self.stop_words or
                self.is_noise_pattern(token.text) or
                token.pos_ not in self.meaningful_pos):
                continue
            lemma = token.lemma_.lower().strip()
            if lemma in company_terms_to_filter or len(lemma) < 3:
                continue
            valid_tokens.append(lemma)
            unigrams.append(lemma)
        for i in range(len(valid_tokens) - 1):
            bigram = f"{valid_tokens[i]} {valid_tokens[i+1]}"
            bigram_contains_company = any(
                company_term in bigram for company_term in company_terms_to_filter
            )
            if not bigram_contains_company:
                bigrams.append(bigram)
        return unigrams, bigrams

    def is_noise_pattern(self, word):
        for pattern in self.noise_patterns:
            if re.match(pattern, word, re.IGNORECASE):
                return True
        return False

    def detect_anomalies_dual_corpus(self, target_job, specialist_corpus, general_corpus, 
                                    job_title="", company_name="", industry=""):
        if not target_job:
            return {"specialist_anomalies": [], "industry_markers": [], "metadata": {}}
        company_terms = self.extract_company_terms(company_name)
        target_unigrams, target_bigrams = self.advanced_text_preprocessing(
            target_job, company_terms
        )
        specialist_unigrams = []
        specialist_bigrams = []
        for job in specialist_corpus:
            if job and not pd.isna(job):
                uni, bi = self.advanced_text_preprocessing(job)
                specialist_unigrams.extend(uni)
                specialist_bigrams.extend(bi)
        general_unigrams = []
        general_bigrams = []
        for job in general_corpus:
            if job and not pd.isna(job):
                uni, bi = self.advanced_text_preprocessing(job)
                general_unigrams.extend(uni)
                general_bigrams.extend(bi)
        specialist_anomalies = self.calculate_anomalies_enhanced(
            target_unigrams, target_bigrams,
            specialist_unigrams, specialist_bigrams,
            job_title, company_name, comparison_type="specialist"
        )
        industry_markers = self.calculate_anomalies_enhanced(
            target_unigrams, target_bigrams,
            general_unigrams, general_bigrams,
            job_title, company_name, comparison_type="industry"
        )
        return {
            "specialist_anomalies": specialist_anomalies,
            "industry_markers": industry_markers,
            "metadata": {
                "industry": industry,
                "specialist_corpus_size": len(specialist_corpus),
                "general_corpus_size": len(general_corpus),
                "target_unigrams_count": len(target_unigrams),
                "target_bigrams_count": len(target_bigrams),
                "company_terms_filtered": len(company_terms)
            }
        }

    def calculate_anomalies_enhanced(self, target_unigrams, target_bigrams, 
                                    corpus_unigrams, corpus_bigrams,
                                    job_title, company_name, comparison_type):
        results = []
        for word_type, target_words, corpus_words in [
            ("unigrams", target_unigrams, corpus_unigrams),
            ("bigrams", target_bigrams, corpus_bigrams)
        ]:
            target_freq = Counter(target_words)
            corpus_freq = Counter(corpus_words)
            target_total = len(target_words)
            corpus_total = len(corpus_words)
            if target_total == 0 or corpus_total == 0:
                continue
            for word, count in target_freq.items():
                target_ratio = count / target_total
                corpus_count = corpus_freq.get(word, 0)
                min_corpus_freq = 1 if word_type == "bigrams" else 1
                if corpus_count < min_corpus_freq:
                    corpus_ratio = 0.0001
                else:
                    corpus_ratio = corpus_count / corpus_total
                anomaly_ratio = target_ratio / corpus_ratio if corpus_ratio > 0 else 999
                if comparison_type == "specialist":
                    min_target_freq = 0.008 if word_type == "bigrams" else 0.004
                    min_anomaly_ratio = 1.2 if word_type == "bigrams" else 1.5
                else:
                    min_target_freq = 0.01 if word_type == "bigrams" else 0.006
                    min_anomaly_ratio = 2.0 if word_type == "bigrams" else 3.0
                if (target_ratio >= min_target_freq and 
                    anomaly_ratio >= min_anomaly_ratio and
                    count >= 1):
                    quality_score = self.calculate_quality_score_enhanced(
                        word, job_title, word_type, comparison_type
                    )
                    if quality_score > 0:
                        results.append({
                            'word': str(word),
                            'anomaly_ratio': float(round(anomaly_ratio, 2)),
                            'target_frequency': float(round(target_ratio * 100, 3)),
                            'corpus_frequency': float(round(corpus_ratio * 100, 4)),
                            'target_count': int(count),
                            'corpus_count': int(corpus_count),
                            'quality_score': float(quality_score),
                            'comparison_type': str(comparison_type),
                            'word_type': str(word_type)
                        })
        results.sort(key=lambda x: (x['quality_score'], x['anomaly_ratio']), reverse=True)
        return results[:8]

    def calculate_quality_score_enhanced(self, word, job_title, word_type, comparison_type):
        score = 1
        if self.is_generic_word(word):
            score -= 3
            logger.debug(f"Penalizing generic word: {word} (low IDF)")
        if job_title and word.lower() in job_title.lower():
            score += 2
        tech_indicators = [
            'python', 'java', 'javascript', 'react', 'vue', 'angular', 'node',
            'aws', 'azure', 'docker', 'kubernetes', 'git', 'sql', 'nosql',
            'machine', 'learning', 'ai', 'data', 'analytics', 'science',
            'devops', 'agile', 'scrum', 'api', 'rest', 'graphql',
            'testing', 'automation', 'ci/cd', 'jenkins', 'terraform'
        ]
        if any(tech in word.lower() for tech in tech_indicators):
            score += 2
        industry_terms = [
            'esg', 'sustainable', 'proptech', 'fintech', 'blockchain', 'cryptocurrency',
            'healthcare', 'medical', 'pharmaceutical', 'biotech',
            'automotive', 'manufacturing', 'logistics', 'supply',
            'consulting', 'strategy', 'framework', 'case study',
            'legal', 'compliance', 'regulatory', 'litigation',
            'öffentlichkeitsarbeit', 'kommunikation', 'leadership communications',
            'international', 'global', 'cross border'
        ]
        if any(term in word.lower() for term in industry_terms):
            score += 2
        if comparison_type == "industry":
            score += 1
        if word_type == "bigrams":
            score += 1
        if re.search(r'\d', word):
            score += 0.5
        explicit_generic = [
            'digital', 'student', 'kreativ', 'innovative', 'modern',
            'new', 'current', 'future', 'excellent', 'strong', 'good',
            'various', 'different', 'multiple', 'general', 'basic'
        ]
        if any(generic in word.lower() for generic in explicit_generic):
            score -= 2
        return max(0, score)

