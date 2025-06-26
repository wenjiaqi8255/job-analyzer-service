import logging
import re
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class LegacyAnomalyDetector:
    def __init__(self, text_preprocessor):
        self.text_preprocessor = text_preprocessor

    def calculate_global_idf(self, descriptions: list[str]) -> dict:
        logger.info("Calculating global IDF from descriptions...")
        all_texts = [text for text in descriptions if text and len(text.strip()) > 10]
        if len(all_texts) < 5:
            logger.warning("Not enough documents to calculate IDF, skipping.")
            return {}

        vectorizer = TfidfVectorizer(
            max_features=20000,
            stop_words=list(self.text_preprocessor.stop_words),
            ngram_range=(1, 2),
            min_df=3,
            token_pattern=r'(?u)\b[\w-]{2,}\b'
        )
        try:
            vectorizer.fit(all_texts)
            idf_dict = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
            self.text_preprocessor.global_idf_cache = idf_dict
            logger.info(f"IDF calculation complete. Found {len(idf_dict)} terms.")
            return idf_dict
        except Exception as e:
            logger.error(f"IDF calculation failed: {e}", exc_info=True)
            return {}

    def detect_anomalies_dual_corpus(self, target_job, specialist_corpus, general_corpus,
                                    job_title="", company_name="", industry=""):
        if not target_job:
            return {"specialist_anomalies": [], "industry_markers": [], "metadata": {}}

        company_terms = self.text_preprocessor.extract_company_terms(company_name)
        target_unigrams, target_bigrams = self.text_preprocessor.advanced_text_preprocessing(target_job, company_terms)

        def get_corpus_grams(corpus):
            unigrams, bigrams = [], []
            for job in corpus:
                if job and not pd.isna(job):
                    uni, bi = self.text_preprocessor.advanced_text_preprocessing(job)
                    unigrams.extend(uni)
                    bigrams.extend(bi)
            return unigrams, bigrams

        specialist_unigrams, specialist_bigrams = get_corpus_grams(specialist_corpus)
        general_unigrams, general_bigrams = get_corpus_grams(general_corpus)

        return {
            "specialist_anomalies": self._calculate_anomalies(target_unigrams, target_bigrams, specialist_unigrams, specialist_bigrams, job_title, "specialist"),
            "industry_markers": self._calculate_anomalies(target_unigrams, target_bigrams, general_unigrams, general_bigrams, job_title, "industry"),
            "metadata": {"industry": industry, "specialist_corpus_size": len(specialist_corpus), "general_corpus_size": len(general_corpus)}
        }

    def _calculate_anomalies(self, target_unigrams, target_bigrams, corpus_unigrams, corpus_bigrams, job_title, comp_type):
        results = []
        for word_type, target_words, corpus_words in [("unigrams", target_unigrams, corpus_unigrams), ("bigrams", target_bigrams, corpus_bigrams)]:
            target_freq, corpus_freq = Counter(target_words), Counter(corpus_words)
            target_total, corpus_total = len(target_words), len(corpus_words)
            if target_total == 0 or corpus_total == 0:
                continue

            for word, count in target_freq.items():
                target_ratio = count / target_total
                corpus_ratio = corpus_freq.get(word, 0) / corpus_total if corpus_freq.get(word, 0) > (1 if word_type == "bigrams" else 1) else 0.0001
                anomaly_ratio = target_ratio / corpus_ratio if corpus_ratio > 0 else 999
                
                min_target_freq = (0.008 if word_type == "bigrams" else 0.004) if comp_type == "specialist" else (0.01 if word_type == "bigrams" else 0.006)
                min_anomaly_ratio = (1.2 if word_type == "bigrams" else 1.5) if comp_type == "specialist" else (2.0 if word_type == "bigrams" else 3.0)

                if target_ratio >= min_target_freq and anomaly_ratio >= min_anomaly_ratio and count >= 1:
                    quality_score = self._calculate_quality_score(word, job_title, word_type, comp_type)
                    if quality_score > 0:
                        results.append({
                            'word': str(word), 'anomaly_ratio': float(round(anomaly_ratio, 2)),
                            'target_frequency': float(round(target_ratio * 100, 3)), 'corpus_frequency': float(round(corpus_ratio * 100, 4)),
                            'target_count': int(count), 'corpus_count': int(corpus_freq.get(word, 0)),
                            'quality_score': float(quality_score), 'comparison_type': str(comp_type), 'word_type': str(word_type)
                        })
        results.sort(key=lambda x: (x['quality_score'], x['anomaly_ratio']), reverse=True)
        return results[:8]

    def _calculate_quality_score(self, word, job_title, word_type, comparison_type):
        score = 1
        if self.text_preprocessor.is_generic_word(word): score -= 3
        if job_title and word.lower() in job_title.lower(): score += 2
        
        tech_indicators = ['python', 'java', 'react', 'aws', 'docker', 'sql', 'machine', 'learning', 'devops', 'api']
        if any(tech in word.lower() for tech in tech_indicators): score += 2

        industry_terms = ['esg', 'fintech', 'healthcare', 'automotive', 'consulting', 'legal', 'international']
        if any(term in word.lower() for term in industry_terms): score += 2

        if comparison_type == "industry": score += 1
        if word_type == "bigrams": score += 1
        if re.search(r'\d', word): score += 0.5
        
        explicit_generic = ['digital', 'student', 'creative', 'innovative', 'excellent', 'strong', 'good']
        if any(generic in word.lower() for generic in explicit_generic): score -= 2
            
        return max(0, score) 