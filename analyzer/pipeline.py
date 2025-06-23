import logging
import pandas as pd
import json
import hashlib
from datetime import datetime, timezone
from analyzer.model import EnhancedJobAnomalyDetector
from analyzer.inference import write_analysis_result
from extractor.job_data_accessor import JobDataAccessor

logger = logging.getLogger(__name__)

def run_pipeline(mode, jobs_to_analyze_df, idf_corpus_df, supabase=None, output_json_path='sample_data/output.json'):
    detector = EnhancedJobAnomalyDetector(supabase_client=supabase)

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
                
                # The old corpus building is no longer needed for semantic analysis.
                # We now use pre-computed baselines.

                # Dynamically classify the role based on the job description.
                role_baseline_for_job = detector.classify_job_role(job_title=title, job_description=description)
                
                logger.info(f"Analyzing '{title}' with classified role='{role_baseline_for_job}' and industry='{industry}'")

                anomalies = detector.detect_semantic_anomalies(
                    target_job=description,
                    role_baseline_name=role_baseline_for_job,
                    industry_baseline_name=industry
                )
                
                if not anomalies:
                    logger.info(f"No semantic anomalies found for {company} - {title}")

                result = {
                    'semantic_anomalies': anomalies,
                    'industry': industry,
                    'company_name': company,
                    'job_title': title,
                    'effective_description': description,
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
