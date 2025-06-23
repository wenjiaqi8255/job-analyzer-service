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
from .text_preprocessor import TextPreprocessor
from .role_classifier import RoleClassifier
from .semantic_anomaly_detector import SemanticAnomalyDetector
from .legacy_anomaly_detector import LegacyAnomalyDetector

import logging
logger = logging.getLogger(__name__)

class EnhancedJobAnomalyDetector:
    def __init__(self, supabase_client: Client = None):
        self.supabase_client = supabase_client
        self.nlp = self._load_spacy_model()
        self.embedding_model, self.embedding_model_name = self._load_embedding_model()
        self.device = self.embedding_model.device if self.embedding_model else "cpu"
        
        self.semantic_baselines = {}
        if self.supabase_client and self.embedding_model:
            self._load_baselines_from_db()

        # Initialize modular components
        self.text_preprocessor = TextPreprocessor(self.nlp)
        self.role_classifier = RoleClassifier(self.embedding_model, self.semantic_baselines, self.text_preprocessor)
        self.semantic_anomaly_detector = SemanticAnomalyDetector(self.embedding_model, self.semantic_baselines, self.text_preprocessor, self.nlp)
        self.legacy_anomaly_detector = LegacyAnomalyDetector(self.text_preprocessor)

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

        self.role_title_embeddings = {}
        self._precompute_role_title_embeddings()

    def _precompute_role_title_embeddings(self):
        """预计算所有role的标准title embeddings并缓存"""
        if not self.embedding_model:
            logger.warning("Embedding model not available, skipping title embeddings precomputation")
            return
            
        logger.info("预计算role title embeddings...")
        
        # 从实际的role baselines生成title embeddings
        role_baselines = self.semantic_baselines.get("role", {})
        if not role_baselines:
            logger.warning("No role baselines available for title embedding precomputation")
            return
        
        # 为每个实际存在的role生成title描述
        role_titles_to_encode = []
        role_names = []
        
        for role_baseline_name in role_baselines.keys():
            # 从baseline name推导title：去掉_baseline后缀，转换为可读标题
            role_key = role_baseline_name.replace('_baseline', '')
            
            # 智能转换role key到title
            title = self._role_key_to_title(role_key)
            role_titles_to_encode.append(title)
            role_names.append(role_baseline_name)  # 使用完整的baseline name作为key
        
        try:
            # 批量编码所有role titles
            title_embeddings = self.embedding_model.encode(role_titles_to_encode, convert_to_tensor=True)
            
            # 存储到缓存，key使用完整的baseline name
            for role_name, title_embedding in zip(role_names, title_embeddings):
                self.role_title_embeddings[role_name] = title_embedding
                
            logger.info(f"✅ 成功预计算 {len(self.role_title_embeddings)} 个role title embeddings")
            logger.debug(f"缓存的role keys: {list(self.role_title_embeddings.keys())}")
            
        except Exception as e:
            logger.error(f"预计算role title embeddings失败: {e}")
            self.role_title_embeddings = {}
    
    def _role_key_to_title(self, role_key: str) -> str:
        """智能转换role key到自然语言title"""
        # 替换下划线为空格，首字母大写
        title = role_key.replace('_', ' ').title()
        
        # 特殊处理一些缩写和术语
        replacements = {
            'Ui Ux': 'UI UX',
            'Ai Ml': 'AI ML', 
            'It': 'IT',
            'Seo': 'SEO',
            'Api': 'API',
            'Devops': 'DevOps',
            'Qa': 'QA'
        }
        
        for old, new in replacements.items():
            title = title.replace(old, new)
            
        return title

    def _preprocess_job_title(self, job_title: str) -> str:
        """简单清理job title，去掉噪音词汇"""
        if not job_title or not job_title.strip():
            return ""
        
        title = job_title.strip()
        
        # 去掉资历级别词汇
        seniority_patterns = [
            r'\b(senior|sr\.?|lead|principal|staff|chief|head of|director of)\b',
            r'\b(junior|jr\.?|entry level|graduate|trainee|intern)\b', 
            r'\b(mid|middle|intermediate|associate)\b'
        ]
        
        for pattern in seniority_patterns:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        
        # 去掉修饰词
        modifier_patterns = [
            r'\b(experienced|skilled|talented|expert|professional)\b',
            r'\b(full time|part time|freelance|contract|remote)\b',
            r'\b(m/f/d|w/m/d|\(m/f/d\)|\(w/m/d\))\b'  # 德语性别标识
        ]
        
        for pattern in modifier_patterns:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        
        # 标准化分隔符和缩写
        title = re.sub(r'[-/&]', ' ', title)  # 统一分隔符
        title = re.sub(r'\bdev\b', 'developer', title, flags=re.IGNORECASE)
        title = re.sub(r'\beng\b', 'engineer', title, flags=re.IGNORECASE) 
        title = re.sub(r'\bmgr\b', 'manager', title, flags=re.IGNORECASE)
        
        # 清理多余空格
        title = re.sub(r'\s+', ' ', title).strip()
        
        logger.debug(f"Title preprocessing: '{job_title}' → '{title}'")
        return title

    def classify_job_role(self, job_title: str = "", job_description: str = "", 
                          min_avg_similarity=0.35, title_weight=0.7) -> str:
        """
        使用title + description的加权融合进行角色分类
        
        Args:
            job_title: 职位标题
            job_description: 职位描述  
            min_avg_similarity: 最小相似度阈值
            title_weight: title的权重 (建议0.7)
        
        Returns:
            角色分类字符串或"general"
        """
        return self.role_classifier.classify_job_role(job_title, job_description, min_avg_similarity, title_weight)

    def _load_baselines_from_db(self):
        logger.info("Loading semantic baselines from database...")
        baselines = fetch_active_semantic_baselines(self.supabase_client)
        if not baselines:
            logger.warning("No baselines loaded from DB.")
            return

        self.semantic_baselines = {"role": {}, "industry": {}, "global": {}}
        for baseline in baselines:
            baseline_id = baseline['id']
            name = baseline['name']
            baseline_type = baseline['baseline_type']
            
            # Simplified data extraction
            source_keywords = [kw for key, value in baseline.get('baseline_data', {}).items() if isinstance(value, list) for kw in value]
            if not source_keywords:
                continue

            vectors_data = fetch_baseline_vectors(self.supabase_client, baseline_id, self.embedding_model_name)
            if not vectors_data:
                continue
            
            # Handle string-encoded JSON
            vectors_dict = json.loads(vectors_data) if isinstance(vectors_data, str) else vectors_data
            if not isinstance(vectors_dict, dict):
                continue
            
            matched_keywords = [kw for kw in source_keywords if kw in vectors_dict]
            if not matched_keywords:
                continue
                
            try:
                vectors = torch.tensor([vectors_dict[kw] for kw in matched_keywords], dtype=torch.float).to(self.device)
                if baseline_type in self.semantic_baselines:
                    self.semantic_baselines[baseline_type][name] = {"keywords": matched_keywords, "vectors": vectors}
            except Exception as e:
                logger.error(f"[{name}] Error creating tensor: {e}", exc_info=True)

        logger.info("Finished loading baselines.")

    def classify_job_industry(self, company_name, job_title, description):
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

    def detect_semantic_anomalies(self, target_job: str, role_baseline_name: str, industry_baseline_name: str = None, similarity_threshold=0.5):
        """
        Detects semantic anomalies using vector similarity against pre-defined baselines.
        """
        return self.semantic_anomaly_detector.detect_semantic_anomalies(target_job, role_baseline_name, industry_baseline_name, similarity_threshold)

    def detect_anomalies_dual_corpus(self, target_job, specialist_corpus, general_corpus, 
                                    job_title="", company_name="", industry=""):
        return self.legacy_anomaly_detector.detect_anomalies_dual_corpus(target_job, specialist_corpus, general_corpus, job_title, company_name, industry)

    def calculate_global_idf(self, descriptions: list[str]) -> dict:
        """
        Calculates global IDF values from a list of job descriptions.
        """
        return self.legacy_anomaly_detector.calculate_global_idf(descriptions)

    def _load_spacy_model(self):
        try:
            return spacy.load("de_core_news_sm")
        except OSError:
            logger.warning("German spaCy model not found, trying English model.")
            try:
                return spacy.load("en_core_web_sm")
            except OSError:
                logger.error("Please install a spaCy model: python -m spacy download de_core_news_sm")
                raise

    def _load_embedding_model(self):
        try:
            model_name = 'all-MiniLM-L6-v2'
            model = SentenceTransformer(model_name)
            logger.info(f"Loaded sentence-transformer model '{model_name}'.")
            return model, model_name
        except Exception as e:
            logger.error(f"Failed to load sentence-transformer model: {e}", exc_info=True)
            return None, None

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

