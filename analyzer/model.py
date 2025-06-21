import re
import pandas as pd
import numpy as np
import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from extractor.industry_keyword_library import INDUSTRY_KEYWORD_LIBRARY

import logging
logger = logging.getLogger(__name__)

class EnhancedJobAnomalyDetector:
    def __init__(self):
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

