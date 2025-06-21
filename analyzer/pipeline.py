import logging
import pandas as pd
import json
import hashlib
from datetime import datetime, timezone
from analyzer.model import EnhancedJobAnomalyDetector
from analyzer.inference import write_analysis_result
from keybert import KeyBERT
from extractor.job_data_accessor import JobDataAccessor

logger = logging.getLogger(__name__)

def _enhance_text_with_keybert(text, kw_model):
    """Uses KeyBERT to extract key phrases and returns them as a single string."""
    if not text or pd.isna(text) or len(text.strip()) < 50:
        return text  # Return original text if it's too short
    try:
        # Extract keywords/keyphrases
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3), 
            stop_words='english',
            use_mmr=True, 
            diversity=0.5,
            top_n=15
        )
        # The result is a list of tuples (phrase, score)
        key_phrases = [kw[0] for kw in keywords]
        return " ".join(key_phrases)
    except Exception as e:
        logger.warning(f"KeyBERT enhancement failed for a text snippet. Returning original text. Error: {e}")
        return text

def run_pipeline(mode, jobs_to_analyze_df, idf_corpus_df, supabase=None, output_json_path='sample_data/output.json', use_keybert: bool = True):
    detector = EnhancedJobAnomalyDetector()
    kw_model = None
    if use_keybert:
        logger.info("KeyBERT enhancement is enabled.")
        kw_model = KeyBERT(model='all-MiniLM-L6-v2')
    else:
        logger.info("KeyBERT enhancement is disabled.")

    if len(jobs_to_analyze_df) == 0:
        logger.warning("No jobs to process")
        return []
    
    logger.info(f"Starting {mode} mode with {len(jobs_to_analyze_df)} jobs to analyze, using a corpus of {len(idf_corpus_df)} jobs for IDF.")
    
    if mode == "train":
        logger.info("Training mode not implemented yet.")
        return None
    elif mode == "inference":
        # Create an 'effective_description' column that prioritizes translated descriptions
        logger.info("Preparing effective descriptions for corpus and analysis jobs...")
        idf_corpus_df['effective_description'] = idf_corpus_df.apply(JobDataAccessor.get_effective_description, axis=1)
        jobs_to_analyze_df['effective_description'] = jobs_to_analyze_df.apply(JobDataAccessor.get_effective_description, axis=1)

        # 1. Calculate IDF on-the-fly from the effective corpus descriptions
        logger.info("Calculating IDF values from corpus...")
        corpus_descriptions = idf_corpus_df['effective_description'].dropna().tolist()
        if corpus_descriptions:
            detector.calculate_global_idf(corpus_descriptions)
            logger.info("IDF calculation complete.")
        else:
            logger.warning("Corpus for IDF is empty, skipping IDF calculation.")

        # Industry classification is disabled as per request.
        # The 'industry' from the source data will be used directly.

        results = []
        error_count = 0
        
        # 3. Process each job
        for idx, row in jobs_to_analyze_df.iterrows():
            try:
                job_id = row['id']
                company = row.get('company_name', 'Unknown')
                title = row.get('job_title', 'Unknown')
                description = row.get('effective_description', '')
                industry = row.get('industry', 'general')

                # Enhance the target job description with KeyBERT if enabled
                if use_keybert and kw_model:
                    enhanced_description = _enhance_text_with_keybert(description, kw_model)
                else:
                    enhanced_description = description
                
                # Build specialist and general corpuses from effective descriptions
                specialist_corpus = idf_corpus_df[idf_corpus_df['industry'] == industry]['effective_description'].dropna().tolist()
                general_corpus = idf_corpus_df[idf_corpus_df['industry'] != industry]['effective_description'].dropna().tolist()

                if len(specialist_corpus) < 2 or len(general_corpus) < 2:
                    logger.warning(f"Skipping {company} - {title}: not enough corpus for industry '{industry}' (specialist: {len(specialist_corpus)}, general: {len(general_corpus)})")
                    continue
                
                anomalies = detector.detect_anomalies_dual_corpus(
                    enhanced_description, specialist_corpus, general_corpus,
                    job_title=title, company_name=company, industry=industry
                )
                
                result = {
                    'dual_corpus_anomalies': anomalies,
                    'industry': industry,
                    'company_name': company,
                    'job_title': title,
                    'effective_description': enhanced_description,
                }
                results.append(result)
                
                if supabase:
                    write_analysis_result(job_id, result, supabase)

            except Exception as e:
                error_count += 1
                logger.error(f"Error processing job {row.get('id', idx)}: {e}", exc_info=True)
                if error_count > max(5, len(jobs_to_analyze_df) * 0.3):
                    logger.error("Error rate exceeded threshold, stopping pipeline.")
                    break
        
        # Local testing output
        if supabase is None:
            output_data = [{**res, 'job_id': jobs_to_analyze_df.iloc[i]['id']} for i, res in enumerate(results)]
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {output_json_path}")
            
        return results
    else:
        raise ValueError(f"Unknown mode: {mode}")
